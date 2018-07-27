import torch
from collections import Counter
from heapq import heappush, heappop, heapify
from bitarray import bitarray


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


# TODO: fixed point arg
class HuffmanParam(object):
    """

    """
    def __init__(self, param=None, is_encode_indices=False, max_bit_length_zero_run_length=4):
        """

        :param param:
        :param is_encode_indices:
        :param max_bit_length_zero_run_length:
        """
        if max_bit_length_zero_run_length <= 0:
            is_encode_indices = False
            max_bit_length_zero_run_length = 0
        self.max_bit_length_zero_run_length = max_bit_length_zero_run_length
        self.is_encode_indices = is_encode_indices
        self.max_zero_run_length = max_zero_run_length = 2 ** max_bit_length_zero_run_length - 2

        assert torch.is_tensor(param)
        self.num_el = num_el = param.numel()
        self.shape = param.size()
        self.bit_stream = {'param': None, 'index': None}
        self.codebook = dict()

        if torch.is_tensor(param):
            if is_encode_indices:
                param = param.view(num_el)
                nonzero_indices = param.nonzero()
                nonzero_indices = torch.sort(nonzero_indices.view(nonzero_indices.numel()))
                run_length = (nonzero_indices[1:] - nonzero_indices[:-1] - 1).cpu().tolist()
                run_length.insert(0, nonzero_indices[0].cpu().tolist())
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
                bit_stream_index = ''
                bit_format = '{:0%db}' % max_bit_length_zero_run_length
                for rl in run_length_chunks:
                    bit_stream_index += bit_format.format(rl)
                self.bit_stream['index'] = bitarray(bit_stream_index)
                param = param[param != 0]
            else:
                self.bit_stream['index'] = None
                param = param.view(num_el)
            # get param codebook
            param_list = param.cpu().tolist()
            symb2freq = Counter(param_list)
            self.codebook = get_huffman_codebook(symb2freq)
            # encode param
            self.bit_stream['param'] = bitarray().encode(self.codebook, param_list)

    @property
    def bit_length(self):
        """

        :return:
        """
        if self.is_encode_indices:
            return 32 * len(self.codebook) + sum(map(lambda v: len(v), self.codebook.values())) + \
               len(self.bit_stream['param']) + len(self.bit_stream['index'])
        else:
            return 32 * len(self.codebook) + sum(map(lambda v: len(v), self.codebook.values())) + \
               len(self.bit_stream['param'])

    @property
    def stats(self):
        """

        :return:
        """
        stats = dict()
        stats['bit_length']['codebook'] = 32 * len(self.codebook) + \
            sum(map(lambda v: len(v), self.codebook.values()))
        stats['bit_length']['param'] = len(self.bit_stream['param'])
        if self.is_encode_indices:
            stats['bit_length']['index'] = len(self.bit_stream['index'])
        else:
            stats['bit_length']['index'] = 0
        stats['compression_ratio'] = (stats['bit_length']['codebook'] +
                                      stats['bit_length']['param'] + stats['bit_length']['index']) / \
                                     (32 * self.num_el)
        stats['num_el'] = self.num_el
        stats['shape'] = self.shape
        return stats

    @property
    def data(self):
        """

        :return:
        """
        if self.is_encode_indices:
            param_nz_list = self.bit_stream['param'].decode(self.codebook)
            bit_stream_index = self.bit_stream['index'].to01()
            param_list = []
            nz_idx = 0
            for i in range(0, len(bit_stream_index), self.max_bit_length_zero_run_length):
                bits = bit_stream_index[i:(i+self.max_bit_length_zero_run_length)]
                run_length = int(bits, 2)
                if run_length > self.max_zero_run_length:
                    param_list.extend(0 * self.max_zero_run_length)
                else:
                    param_list.extend([0] * run_length)
                    param_list.append(param_nz_list[nz_idx])
                    nz_idx += 1
            param_list.extend([0] * (self.num_el - len(param_list)))
        else:
            param_list = self.bit_stream['param'].decode(self.codebook)
        return torch.tensor(param_list).view(self.shape)

    def state_dict(self):
        """

        :return:
        """
        state_dict = dict()
        state_dict['is_encode_indices'] = self.is_encode_indices
        state_dict['max_bit_length_zero_run_length'] = self.max_bit_length_zero_run_length
        state_dict['max_zero_run_length'] = self.max_zero_run_length
        state_dict['num_el'] = self.num_el
        state_dict['shape'] = self.shape
        state_dict['bit_stream'] = self.bit_stream
        state_dict['codebook'] = self.codebook

    def load_state_dict(self, state_dict):
        """

        :param state_dict:
        :return:
        """
        for k, v in state_dict.items():
            self.__setattr__(k, v)
        return self


class HuffmanCodec(object):
    """

    """
    