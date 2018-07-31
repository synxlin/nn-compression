import math
import torch
import multiprocessing
from collections import Counter
from heapq import heappush, heappop, heapify
from bitarray import bitarray
from itertools import repeat

from ..replicate import replicate
from ..utils import iter_str_every


def get_huffman_codebook(symb2freq):
    """
    Huffman encode the given dict mapping symbols to weights
    :param symb2freq: dict, {symbol: frequency}
    :return:
        dict, value(float/int) : code(bitarray)
    """
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    codebook = sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    return dict(map(lambda x: (x[0], bitarray(x[1])), codebook))


def get_vanilla_codebook(symb):
    codebook = dict()
    symb = set(symb)
    bit_length = int(math.ceil(math.log(len(symb), 2)))
    bit_format = '{:0%db}' % bit_length
    for i, s in enumerate(symb):
        codebook[s] = bitarray(bit_format.format(i))
    return codebook


def _encode_positive_integer(args):
    """

    :param args[0]: x: int
    :param args[1]: bit_length: int, bit length of fixed point x,
                                     including sign bit
    :return:
        str,
    """
    bit_format = '{:0%db}' % args[1]
    return bit_format.format(args[0])


def _encode_float(args):
    """

    :param args[0]: x: int or float
    :param args[1]: bit_length: int, bit length of fixed point x,
                                     including sign bit
    :param args[2]: bit_length_integer: int, bit length of integer part
                                             of fixed point x
    :return:
        str,
    """
    bit_format = '{:0%db}' % args[1]
    mul_coeff = 2 ** (args[1] - args[2] - 1)
    add_coeff = 2 ** args[1]
    return bit_format.format(int(math.floor(args[0] * mul_coeff)) + add_coeff)[-args[1]:]


def _decode_positive_integer(bits):
    """

    :param bits:
    :return:
    """
    return int(bits, 2)


def _decode_float(args):
    """

    :param args[0]: bits:
    :param args[1]: bit_length:
    :param args[2]: bit_length_integer:
    :return:
    """
    div_coeff = 2 ** (args[2] - args[1] + 1)
    max_coeff = 2 ** (args[1] - 1)
    sub_coeff = 2 * max_coeff

    num = int(args[0], 2)
    if num >= max_coeff:
        num -= sub_coeff
    num *= div_coeff
    return num


class EncodedParam(object):

    def __init__(self, param=None, method='huffman',
                 bit_length=8, bit_length_integer=0,
                 encode_indices=False, bit_length_zero_run_length=4):
        """
        EncodedParam class
        :param param: torch.(cuda.)tensor, default=None
        :param method: str, coding method,
                            choose from ['vanilla', 'fixed_point', 'huffman']
        :param bit_length: int, bit length of fixed point param,
                                including sign bit, default=8
        :param bit_length_integer: int, bit length of integer part
                                        of fixed point param, default=0
        :param encode_indices: bool, whether to encode zero run length, default=False
        :param bit_length_zero_run_length: int, bit length of zero run length
                                                without sign bit
                                                since run length is non-negative
        """
        assert method in ['vanilla', 'fixed_point', 'huffman']
        if bit_length <= 0:
            bit_length = 0
            bit_length_integer = 0
            if method == 'fixed_point':
                method = 'vanilla'
        if bit_length_integer < 0 or method != 'fixed_point':
            bit_length_integer = 0
        self.method = method
        self.bit_length = bit_length
        self.bit_length_integer = bit_length_integer
        if bit_length_zero_run_length <= 0:
            encode_indices = False
            bit_length_zero_run_length = 0
        self.max_bit_length_zero_run_length = bit_length_zero_run_length
        self.encode_indices = encode_indices
        self.max_zero_run_length = max_zero_run_length = 2 ** bit_length_zero_run_length - 2

        self.bit_stream = {'param': None, 'index': None}
        self.codebook = None

        if torch.is_tensor(param):
            self.num_el = num_el = param.numel()
            self.num_nz = self.num_el
            self.shape = param.size()
            num_cpu = multiprocessing.cpu_count()
            if encode_indices:
                param = param.view(num_el)
                nonzero_indices = param.nonzero()
                self.num_nz = nonzero_indices.numel()
                nonzero_indices, _ = torch.sort(nonzero_indices.view(self.num_nz))
                nonzero_indices[1:] -= (nonzero_indices[:-1] + 1)
                run_length = nonzero_indices.cpu().tolist()
                num_chunks = 0
                run_length_chunks = []
                for rl in run_length:
                    if rl <= max_zero_run_length:
                        run_length_chunks.append(rl)
                    else:
                        left_rl = rl
                        while left_rl > max_zero_run_length:
                            run_length_chunks.append(max_zero_run_length + 1)
                            left_rl -= max_zero_run_length
                            num_chunks += 1
                        run_length_chunks.append(left_rl)
                # encode indices (fixed point)
                # bit_format = '{:0%db}' % bit_length_zero_run_length
                # parallel encode
                pool = multiprocessing.Pool(processes=num_cpu)
                bit_stream_index = ''.join(pool.map(_encode_positive_integer,
                                                    zip(run_length_chunks,
                                                        repeat(bit_length_zero_run_length))))
                pool.close()
                self.bit_stream['index'] = bitarray(bit_stream_index)
                param = param[param != 0]
            else:
                self.bit_stream['index'] = bitarray()
                param = param.view(num_el)
            # get param codebook
            param_list = param.cpu().tolist()
            if self.method == 'huffman':
                symb2freq = Counter(param_list)
                self.codebook = get_huffman_codebook(symb2freq)
                # encode param
                self.bit_stream['param'] = bitarray()
                self.bit_stream['param'].encode(self.codebook, param_list)
            elif self.method == 'vanilla':
                symb = set(param_list)
                self.codebook = get_vanilla_codebook(symb)
                # encode param
                self.bit_stream['param'] = bitarray()
                self.bit_stream['param'].encode(self.codebook, param_list)
            else:  # fixed point
                # bit_format = '{:0%db}' % self.bit_length
                # mul_coeff = 2 ** (self.bit_length - self.bit_length_integer - 1)
                # add_coeff = 2 ** self.bit_length
                # bit_stream = ''.join(map(lambda x: bit_format
                #                          .format(int(math.floor(x * mul_coeff)) + add_coeff)[-self.bit_length:],
                #                          param_list))
                # parallel encode
                pool = multiprocessing.Pool(processes=num_cpu)
                bit_stream = ''.join(pool.map(_encode_float,
                                              zip(param_list, repeat(bit_length),
                                                  repeat(bit_length_integer))))
                pool.close()
                self.bit_stream['param'] = bitarray(bit_stream)
        else:
            self.num_el = 0
            self.num_nz = 0
            self.shape = None

    @property
    def memory_size(self):
        """
        memory size in bit (total bit length) after encoding
        :return:
            int, bit length after encoding
        """
        if self.codebook is None:
            return len(self.bit_stream['param']) + len(self.bit_stream['index'])
        else:
            return 32 * len(self.codebook) + \
                   sum(map(lambda v: len(v), self.codebook.values())) + \
                   len(self.bit_stream['param']) + len(self.bit_stream['index'])

    @property
    def stats(self):
        """
        stats of encoding
        :return:
            dict, containing info of memory_size of codebook/param/index, compression ratio, num_el and shape
        """
        stats = dict()
        stats['memory_size'] = dict()
        stats['bit_length'] = dict()

        if self.codebook is None:
            stats['memory_size']['codebook'] = 0
            stats['bit_length']['codebook'] = 0
        else:
            stats['bit_length']['codebook'] = sum(map(lambda v: len(v), self.codebook.values()))
            stats['memory_size']['codebook'] = 32 * len(self.codebook) + stats['bit_length']['codebook']
            stats['bit_length']['codebook'] /= len(self.codebook)

        stats['memory_size']['param'] = len(self.bit_stream['param'])
        stats['bit_length']['param'] = stats['memory_size']['param'] / self.num_nz

        stats['memory_size']['index'] = len(self.bit_stream['index'])
        stats['bit_length']['index'] = stats['memory_size']['index'] / self.num_nz

        stats['memory_size']['total'] = stats['memory_size']['codebook'] + stats['memory_size']['param'] + \
            stats['memory_size']['index']
        stats['compression_ratio'] = (32 * self.num_el) / stats['memory_size']['total']

        stats['num_el'] = self.num_el
        stats['num_nz'] = self.num_nz
        stats['shape'] = self.shape
        return stats

    @property
    def data(self):
        """
        returns decoded param
        :return: torch.tensor, param
        """
        if self.codebook is None:
            bit_stream = self.bit_stream['param'].to01()
            bit_length = self.bit_length
            bit_length_integer = self.bit_length_integer

            # div_coeff = 2 ** (self.bit_length_integer - bit_length + 1)
            # max_coeff = 2 ** (bit_length - 1)
            # sub_coeff = 2 * max_coeff
            # param_list = []
            # for i in range(0, len(bit_stream), bit_length):
            #     bits = bit_stream[i:(i + bit_length)]
            #     num = int(bits, 2)
            #     if num >= max_coeff:
            #         num -= sub_coeff
            #     num *= div_coeff
            #     param_list.append(num)

            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            param_list = list(pool.map(_decode_float,
                                       zip(iter_str_every(bit_stream, bit_length),
                                           repeat(bit_length), repeat(bit_length_integer))))
            pool.close()
        else:
            param_list = self.bit_stream['param'].decode(self.codebook)

        if self.encode_indices:
            param_nz_list = param_list
            bit_stream = self.bit_stream['index'].to01()
            param_list = []
            nz_idx = 0
            for i in range(0, len(bit_stream), self.max_bit_length_zero_run_length):
                bits = bit_stream[i:(i+self.max_bit_length_zero_run_length)]
                run_length = int(bits, 2)
                if run_length > self.max_zero_run_length:
                    param_list.extend([0] * self.max_zero_run_length)
                else:
                    param_list.extend([0] * run_length)
                    param_list.append(param_nz_list[nz_idx])
                    nz_idx += 1
            param_list.extend([0] * (self.num_el - len(param_list)))

        return torch.tensor(param_list).view(self.shape)

    def state_dict(self):
        """
        Returns a dictionary containing a whole state of the EncodedParam
        :return: dict, a dictionary containing a whole state of the EncodedParam
        """
        state_dict = dict()
        state_dict['method'] = self.method
        state_dict['bit_length'] = self.bit_length
        state_dict['bit_length_integer'] = self.bit_length_integer
        state_dict['encode_indices'] = self.encode_indices
        state_dict['max_bit_length_zero_run_length'] = self.max_bit_length_zero_run_length
        state_dict['max_zero_run_length'] = self.max_zero_run_length
        state_dict['num_el'] = self.num_el
        state_dict['num_nz'] = self.num_nz
        state_dict['shape'] = self.shape
        state_dict['bit_stream'] = self.bit_stream
        state_dict['codebook'] = self.codebook
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Recover EncodedParam
        :param state_dict: dict, a dictionary containing a whole state of the EncodedParam
        :return: EncodedParam
        """
        for k, v in state_dict.items():
            self.__setattr__(k, v)


class EncodedModule(object):

    def __init__(self, module, encoded_param):
        """
        Encoded Module class
        :param module: torch.nn.Module, network model or nn module
        :param encoded_param: dict, {param name(str): encoded parameters(dict, EncodedParam.state_dict())}
        """
        assert isinstance(module, torch.nn.Module)
        assert isinstance(encoded_param, dict)
        self.module = replicate(module)
        self.encoded_param = encoded_param

        for param_name, param in self.module.named_parameters():
            if 'AuxLogits' in param_name or param_name in self.encoded_param:
                param.data.set_()

    def state_dict(self):
        """
        Returns a dictionary containing a whole state of the encoded module
        :return: dict, a dictionary containing a whole state of the encoded module
        """
        state_dict = self.module.state_dict()
        for param_name, param in self.encoded_param.items():
            state_dict[param_name] = param.state_dict()
        return state_dict
