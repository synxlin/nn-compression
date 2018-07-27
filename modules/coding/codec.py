import re
import torch
import math
import numpy as np
from heapq import heappush, heappop, heapify
from collections import Counter
from ..utils import AverageMeter


def huffman_encode(symb2freq):
    """
    Huffman encode the given dict mapping symbols to weights
    :param symb2freq:
    :return:
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
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


def decode(codebook, bitstring):
    rst = []
    while bitstring:
        flag = False
        for k, v in codebook.items():
            if bitstring.startswith(k):
                flag = True
                rst.append(v)
                bitstring = bitstring[len(k):]
                break
        if not flag:
            return rst
    return rst


def quantize_encode(symb):
    codebook = dict()
    symb = set(symb)
    bit_length = math.log(len(symb), 2)
    bit_format = '{:0%db}' % bit_length
    for i, s in enumerate(symb):
        codebook[s] = bit_format.format(i)
    return codebook


class HuffmanParam(object):
    def __init__(self, param=None, is_encode_indices=False, max_zero_run_length_bit=4):
        if max_zero_run_length_bit <= 0:
            is_encode_indices = False
        self.is_encode_indices = is_encode_indices
        if torch.is_tensor(param):
            assert torch.is_tensor(param)
            self.numel = numel = param.numel()
            self.shape = param.size()
            # get param codebook
            param_list = param.view(numel).tolist()
            symb2freq = Counter(param_list)
            self.param_nonzero_num = numel - symb2freq[float(0)]
            if is_encode_indices:
                # not encode 0
                del symb2freq[float(0)]
            huff_param = huffman_encode(symb2freq)
            self.param_encode_book = dict()
            self.param_decode_book = dict()
            self.param_codebook_data_size = 0
            for h in huff_param:
                if is_encode_indices:
                    self.param_encode_book[h[0]] = (h[1], symb2freq[h[0]] / self.param_nonzero_num)
                else:
                    self.param_encode_book[h[0]] = (h[1], symb2freq[h[0]] / numel)
                self.param_decode_book[h[1]] = h[0]
                self.param_codebook_data_size += (len(h[1]) + 32)  # entry bit size + float
            self.param_entropy_bit = sum(-val[1] * math.log(val[1], 2) for _, val in self.param_encode_book.items())
            self.param_huffman_bit = sum(len(val[0]) * val[1] for _, val in self.param_encode_book.items())
            # encode parameters
            if is_encode_indices:
                bitstring = ''.join(self.param_encode_book[p][0] for p in param_list if p != 0)
            else:
                bitstring = ''.join(self.param_encode_book[p][0] for p in param_list)
            self.param_data_size = len(bitstring)
            self.data_size = self.param_codebook_data_size + self.param_data_size
            self.param_compress_ratio = self.data_size / (numel * 32)
            length = int(math.ceil(len(bitstring) / 64) * 64)
            bitstring += '0' * (length - len(bitstring))
            self.huffman_param = [int(bitstring[i:i+64], 2) for i in range(0, len(bitstring), 64)]
            if is_encode_indices:
                # get zero run length
                nz_pos = []
                for i in range(self.numel):
                    if param_list[i] != 0:
                        nz_pos.append(i)
                nz_pos.insert(0, -1)
                nz_pos = np.asarray(nz_pos)
                run_length = (nz_pos[1:] - nz_pos[:-1] - 1).tolist()
                self.max_zero_run_length = max_zero_run_length = 2 ** max_zero_run_length_bit - 2
                chunked_num = 0
                # make run length into small chunks
                chunk_length = []
                for rl in run_length:
                    if rl <= max_zero_run_length:
                        chunk_length.append(rl)
                    else:
                        left_rl = rl
                        while left_rl > max_zero_run_length:
                            chunk_length.append(max_zero_run_length+1)
                            left_rl -= max_zero_run_length
                            chunked_num += 1
                        chunk_length.append(left_rl)
                # huffman indices
                self.index_chunk_num = len(chunk_length)
                symb2freq = Counter(chunk_length)
                huff_indices = huffman_encode(symb2freq)
                self.index_encode_book = dict()
                self.index_decode_book = dict()
                self.index_codebook_data_size = 0
                for h in huff_indices:
                    self.index_encode_book[h[0]] = (h[1], symb2freq[h[0]] / self.index_chunk_num)
                    self.index_decode_book[h[1]] = h[0]
                    self.index_codebook_data_size += (len(h[1]) + max_zero_run_length_bit)
                self.index_huffman_bit = sum(len(val[0]) * val[1] for _, val in self.index_encode_book.items())
                # encode indices
                bitstring = ''.join(self.index_encode_book[p][0] for p in chunk_length)
                self.index_data_size = len(bitstring)
                length = int(math.ceil(len(bitstring) / 64) * 64)
                bitstring += '0' * (length - len(bitstring))
                self.huffman_index = [int(bitstring[i:i + 64], 2) for i in range(0, len(bitstring), 64)]
                self.index_chunked_ratio = chunked_num / self.param_nonzero_num
                self.data_size += (self.index_codebook_data_size + self.index_data_size)
                self.compress_ratio = self.data_size / (numel*32)
            else:
                self.index_encode_book = self.index_decode_book = dict()
                self.index_codebook_data_size = self.index_data_size = 0
                self.max_zero_run_length = self.index_huffman_bit = 0
                self.index_chunk_num = self.index_chunked_ratio = 0
                self.huffman_index = []
                self.compress_ratio = self.param_compress_ratio
        else:
            self.is_encode_indices = False
            self.param_encode_book = self.param_decode_book = self.index_encode_book = self.index_decode_book = dict()
            self.param_codebook_data_size = self.param_data_size = self.param_compress_ratio =0
            self.index_codebook_data_size = self.index_data_size = 0
            self.param_huffman_bit = self.param_entropy_bit = 0
            self.max_zero_run_length = self.index_huffman_bit = 0
            self.data_size = self.compress_ratio = 0
            self.numel = self.param_nonzero_num = self.index_chunk_num = self.index_chunked_ratio = 0
            self.shape = (0,)
            self.huffman_param = self.huffman_index = []

    @property
    def data(self):
        bitstring = ''.join('{:064b}'.format(x) for x in self.huffman_param)
        int_list = decode(self.param_decode_book, bitstring)
        int_list = int_list[:self.numel]
        if self.is_encode_indices:
            bitstring = ''.join('{:064b}'.format(x) for x in self.huffman_index)
            chunk_list = decode(self.index_decode_book, bitstring)
            chunk_list = chunk_list[:self.index_chunk_num]
            data_list = []
            nz_idx = 0
            for chunk_length in chunk_list:
                if chunk_length > self.max_zero_run_length:
                    data_list.extend([0] * int(self.max_zero_run_length))
                else:
                    data_list.extend([0] * int(chunk_length))
                    data_list.append(int_list[nz_idx])
                    nz_idx += 1
            data_list.extend([0] * (self.numel - len(data_list)))
            int_list = data_list
        param = torch.FloatTensor(int_list).view(self.shape)
        return param

    def state_dict(self, is_compress=False):
        state_dict = dict()
        if is_compress:
            state_dict['is_encode_indices'] = self.is_encode_indices
            state_dict['huffman_param'] = self.huffman_param
            state_dict['huffman_index'] = self.huffman_index
            state_dict['max_zero_run_length'] = self.max_zero_run_length
            state_dict['param_decode_book'] = self.param_decode_book
            state_dict['index_decode_book'] = self.index_decode_book
            state_dict['numel'] = self.numel
            state_dict['shape'] = self.shape
            state_dict['index_chunk_num'] = self.index_chunk_num
        else:
            for k, v in vars(self).items():
                state_dict[k] = v
        return state_dict

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)
        return self


class HuffmanCoder(object):
    def __init__(self, rule_path):
        content = map(lambda x: x.split(), open(rule_path).readlines())
        content = filter(lambda x: len(x) == 2, content)
        content = list(map(lambda x: (x[0], int(x[1])), content))
        self.rules = content
        print_section('Huffman Coding Rules:', up=89, down=False)
        print_section(self.rules, up=False, down=89)

    def encode_param(self, param, name):
        rule_id = -1
        for idx, x in enumerate(self.rules):
            m = re.match(x[0], name)
            if m is not None and len(name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            huffman_param = HuffmanParam(param, is_encode_indices=True, max_zero_run_length_bit=self.rules[rule_id][1])
            print('{:^30} | {:<25} | {:<25} | {:<25} | {:<25} | {:<25} | '
                  '{:<25} | {:<25} | {:<25} | {:<25} | {:<25} | {:<25}'.
                  format(name, huffman_param.param_huffman_bit, huffman_param.index_huffman_bit,
                         huffman_param.compress_ratio, huffman_param.param_nonzero_num, huffman_param.index_chunk_num,
                         huffman_param.index_chunked_ratio, huffman_param.param_codebook_data_size,
                         huffman_param.index_codebook_data_size, huffman_param.param_data_size,
                         huffman_param.index_data_size, huffman_param.data_size))
            return huffman_param
        else:
            print('{:<30} | skipping'.format(name))
            return None

    def encode(self, network):
        assert isinstance(network, torch.nn.Module)
        from .replicate import replicate
        replica = replicate(network)
        huffman_parameters = dict()
        # statistics
        huffman_overall_compress_ratio = AverageMeter()
        huffman_compress_ratio = AverageMeter()
        huffman_param_bit = AverageMeter()
        huffman_index_bit = AverageMeter()
        huffman_param_codebook_data_size = AverageMeter()
        huffman_index_codebook_data_size = AverageMeter()
        huffman_param_data_size = AverageMeter()
        huffman_index_data_size = AverageMeter()
        huffman_overall_param_data_size = AverageMeter()
        huffman_data_size = AverageMeter()
        huffman_overall_data_size = AverageMeter()
        print_section('Start Huffman Encoding')
        print('{:^30} | {:^25} | {:^25} | {:^25} | {:^25} | {:^25} | '
              '{:^25} | {:^25} | {:^25} | {:^25} | {:^25} | {:^25}'.
              format('name', 'huffman bit', 'huffman index bit', 'compress ratio',
                     'huffman nonzero', 'huffman index num', 'index chunked ratio',
                     'param codebook size', 'index codebook size',
                     'param data size', 'index data size', 'data size'))
        for i, (name, param) in enumerate(replica.named_parameters()):
            if 'AuxLogits' in name:
                continue
            huffman_param = self.encode_param(param=param.data, name=name)
            if huffman_param is not None:
                huffman_parameters[i] = (name, huffman_param)
                assert torch.equal(param.data, huffman_param.data)
                param.data = param.data.new()
                # statistics
                huffman_compress_ratio.update(huffman_param.compress_ratio, huffman_param.numel*32)
                huffman_overall_compress_ratio.update(huffman_param.compress_ratio, huffman_param.numel * 32)
                huffman_param_bit.update(huffman_param.param_huffman_bit, huffman_param.param_nonzero_num)
                huffman_index_bit.update(huffman_param.index_huffman_bit, huffman_param.index_chunk_num)
                huffman_param_codebook_data_size.update(huffman_param.param_codebook_data_size)
                huffman_index_codebook_data_size.update(huffman_param.index_codebook_data_size)
                huffman_param_data_size.update(huffman_param.param_data_size)
                huffman_index_data_size.update(huffman_param.index_data_size)
                huffman_data_size.update(huffman_param.data_size)
                huffman_overall_data_size.update(huffman_param.data_size)
                huffman_overall_param_data_size.update(huffman_param.param_data_size)
            huffman_overall_data_size.update(param.data.numel()*32)
            huffman_overall_param_data_size.update(param.data.numel() * 32)
            huffman_overall_compress_ratio.update(1, param.data.numel() * 32)
        print('Huffman Param Bit | {}'.format(huffman_param_bit.avg))
        print('Huffman Index Bit | {}'.format(huffman_index_bit.avg))
        print('Huffman Compress Ratio | {}'.format(huffman_compress_ratio.avg))
        print('Huffman Overall Compress Ratio | {}'.format(huffman_overall_compress_ratio.avg))
        print('Huffman Param Codebook Size | {}'.format(huffman_param_codebook_data_size.sum))
        print('Huffman Index Codebook Size | {}'.format(huffman_index_codebook_data_size.sum))
        print('Huffman Param Data Size | {}'.format(huffman_param_data_size.sum))
        print('Huffman Index Data Size | {}'.format(huffman_index_data_size.sum))
        print('Huffman Data Size | {}'.format(huffman_data_size.sum))
        print('Huffman Overall Param Data Size | {}'.format(huffman_overall_param_data_size.sum))
        print('Huffman Overall Data Size | {}'.format(huffman_overall_data_size.sum))
        print_section('Stop Huffman Encoding')
        return replica, huffman_parameters

    @staticmethod
    def decode(network, encode_parameters):
        assert isinstance(network, torch.nn.Module)
        print_section('Start Huffman Decoding')

        def _worker(i, name, param, huffman_parameter):
            if name == huffman_parameter[0]:
                print('Decoding ' + name)
                huffman_param = huffman_parameter[1]
                param.data = param.data.new(huffman_param.shape).copy_(huffman_param.data)

        for i, (name, param) in enumerate(network.named_parameters()):
            if i in encode_parameters:
                _worker(i, name, param, encode_parameters[i])
        print_section('Stop Huffman Decoding')


class QuantizeParam(object):
    def __init__(self, param=None, is_encode_indices=False, max_zero_run_length_bit=4):
        if max_zero_run_length_bit <= 0:
            is_encode_indices = False
        self.is_encode_indices = is_encode_indices
        if torch.is_tensor(param):
            assert torch.is_tensor(param)
            self.numel = numel = param.numel()
            self.shape = param.size()
            # get param codebook
            param_list = param.view(numel).tolist()
            symb = set(param_list)
            symb2freq = Counter(param_list)
            self.param_nonzero_num = numel - symb2freq[float(0)]
            quantize_param = quantize_encode(symb)
            self.param_encode_book = dict()
            self.param_decode_book = dict()
            self.param_codebook_data_size = 0
            for k, v in quantize_param.items():
                if is_encode_indices:
                    self.param_encode_book[k] = (v, symb2freq[k] / self.param_nonzero_num)
                else:
                    self.param_encode_book[k] = (v, symb2freq[k] / numel)
                self.param_decode_book[v] = k
                self.param_codebook_data_size += (len(v) + 32)  # entry bit size + float
            self.param_quantize_bit = math.log(len(symb), 2)
            self.param_entropy_bit = sum(-val[1] * math.log(val[1], 2) for _, val in self.param_encode_book.items())
            # encode
            if is_encode_indices:
                # get zero run length
                self.max_zero_run_length = max_zero_run_length = 2 ** max_zero_run_length_bit - 1
                pre_nz_pos = -1
                values = []
                chunk_length = []
                chunked_num = 0
                for i in range(self.numel):
                    if param_list[i] != 0:
                        rl = i - pre_nz_pos - 1
                        left_rl = rl
                        while left_rl > self.max_zero_run_length:
                            chunk_length.append(max_zero_run_length)
                            values.append(0)
                            left_rl -= (max_zero_run_length+1)
                            chunked_num += 1
                        chunk_length.append(left_rl)
                        values.append(param_list[i])
                        pre_nz_pos = i
                self.index_chunk_num = len(chunk_length)
                # first encode parameters
                bitstring = ''.join(self.param_encode_book[p][0] for p in values)
                self.param_data_size = len(bitstring)
                self.data_size = self.param_codebook_data_size + self.param_data_size
                self.param_compress_ratio = self.data_size / (numel * 32)
                length = int(math.ceil(len(bitstring) / 64) * 64)
                bitstring += '0' * (length - len(bitstring))
                self.quantize_param = [int(bitstring[i:i + 64], 2) for i in range(0, len(bitstring), 64)]
                # then encode indices
                self.index_quantize_bit = max_zero_run_length_bit
                bit_format = '{:0%db}' % max_zero_run_length_bit
                bitstring = ''.join(bit_format.format(p) for p in chunk_length)
                self.index_data_size = len(bitstring)
                length = int(math.ceil(len(bitstring) / 64) * 64)
                bitstring += '0' * (length - len(bitstring))
                self.quantize_index = [int(bitstring[i:i + 64], 2) for i in range(0, len(bitstring), 64)]
                self.index_chunked_ratio = chunked_num / self.param_nonzero_num
                self.data_size += self.index_data_size
                self.compress_ratio = self.data_size / (numel * 32)
            else:
                bitstring = ''.join(self.param_encode_book[p][0] for p in param_list)
                self.param_data_size = len(bitstring)
                self.data_size = self.param_codebook_data_size + self.param_data_size
                self.param_compress_ratio = self.data_size / (numel * 32)
                length = int(math.ceil(len(bitstring) / 64) * 64)
                bitstring += '0' * (length - len(bitstring))
                self.quantize_param = [int(bitstring[i:i+64], 2) for i in range(0, len(bitstring), 64)]
                self.index_data_size = self.max_zero_run_length = self.index_quantize_bit = 0
                self.index_chunk_num = self.index_chunked_ratio = 0
                self.quantize_index = []
                self.compress_ratio = self.param_compress_ratio
        else:
            self.is_encode_indices = False
            self.param_encode_book = self.param_decode_book = dict()
            self.param_codebook_data_size = self.param_data_size = self.index_data_size = 0
            self.param_quantize_bit = self.max_zero_run_length = self.index_quantize_bit = 0
            self.data_size = self.compress_ratio = self.param_compress_ratio = 0
            self.numel = self.param_nonzero_num = self.index_chunk_num = self.index_chunked_ratio = 0
            self.shape = (0,)
            self.quantize_param = self.quantize_index = []

    @property
    def data(self):
        bitstring = ''.join('{:064b}'.format(x) for x in self.quantize_param)
        int_list = decode(self.param_decode_book, bitstring)
        int_list = int_list[:self.numel]
        if self.is_encode_indices:
            bitstring = ''.join('{:064b}'.format(x) for x in self.quantize_index)
            chunk_list = [int(bitstring[i:i+self.index_quantize_bit], 2)
                          for i in range(0, len(bitstring), self.index_quantize_bit)]
            chunk_list = chunk_list[:self.index_chunk_num]
            int_list = int_list[:self.index_chunk_num]
            data_list = []
            for chunk_length, value in zip(chunk_list, int_list):
                data_list.extend([0] * int(chunk_length))
                data_list.append(value)
            data_list.extend([0] * int(self.numel - len(data_list)))
            int_list = data_list
        param = torch.FloatTensor(int_list).view(self.shape)
        return param

    def state_dict(self, is_compress=False):
        state_dict = dict()
        if is_compress:
            state_dict['is_encode_indices'] = self.is_encode_indices
            state_dict['quantize_param'] = self.quantize_param
            state_dict['quantize_index'] = self.quantize_index
            state_dict['index_quantize_bit'] = self.index_quantize_bit
            state_dict['param_decode_book'] = self.param_decode_book
            state_dict['numel'] = self.numel
            state_dict['shape'] = self.shape
            state_dict['index_chunk_num'] = self.index_chunk_num
        else:
            for k, v in vars(self).items():
                state_dict[k] = v
        return state_dict

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)
        return self


class QuantizeCoder(object):
    def __init__(self, rule_path):
        content = map(lambda x: x.split(), open(rule_path).readlines())
        content = filter(lambda x: len(x) == 2, content)
        content = list(map(lambda x: (x[0], int(x[1])), content))
        self.rules = content
        print_section('Quantize Coding Rules:', up=89, down=False)
        print_section(self.rules, up=False, down=89)

    def encode_param(self, param, name):
        rule_id = -1
        for idx, x in enumerate(self.rules):
            m = re.match(x[0], name)
            if m is not None and len(name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            quantize_param = QuantizeParam(param, is_encode_indices=True, max_zero_run_length_bit=self.rules[rule_id][1])
            print('{:^30} | {:<25} | {:<25} | {:<25} | {:<25} | {:<25} | '
                  '{:<25} | {:<25} | {:<25} | {:<25} | {:<25}'.
                  format(name, quantize_param.param_quantize_bit, quantize_param.index_quantize_bit,
                         quantize_param.compress_ratio, quantize_param.param_nonzero_num, quantize_param.index_chunk_num,
                         quantize_param.index_chunked_ratio, quantize_param.param_codebook_data_size,
                         quantize_param.param_data_size, quantize_param.index_data_size, quantize_param.data_size))
            return quantize_param
        else:
            print('{:<30} | skipping'.format(name))
            return None

    def encode(self, network):
        assert isinstance(network, torch.nn.Module)
        from .replicate import replicate
        replica = replicate(network)
        quantize_parameters = dict()
        # statistics
        quantize_overall_compress_ratio = AverageMeter()
        quantize_compress_ratio = AverageMeter()
        quantize_param_bit = AverageMeter()
        quantize_index_bit = AverageMeter()
        quantize_param_codebook_data_size = AverageMeter()
        quantize_param_data_size = AverageMeter()
        quantize_index_data_size = AverageMeter()
        quantize_overall_param_data_size = AverageMeter()
        quantize_data_size = AverageMeter()
        quantize_overall_data_size = AverageMeter()
        print_section('Start Quantize Encoding')
        print('{:^30} | {:^25} | {:^25} | {:^25} | {:^25} | {:^25} | '
              '{:^25} | {:^25} | {:^25} | {:^25} | {:^25}'.
              format('name', 'quantize bit', 'quantize index bit', 'compress ratio',
                     'quantize nonzero', 'quantize index num', 'index chunked ratio',
                     'param codebook size', 'param data size', 'index data size', 'data size'))
        for i, (name, param) in enumerate(replica.named_parameters()):
            if 'AuxLogits' in name:
                continue
            quantize_param = self.encode_param(param=param.data, name=name)
            if quantize_param is not None:
                quantize_parameters[i] = (name, quantize_param)
                assert torch.equal(param.data, quantize_param.data)
                param.data = param.data.new()
                # statistics
                quantize_compress_ratio.update(quantize_param.compress_ratio, quantize_param.numel*32)
                quantize_overall_compress_ratio.update(quantize_param.compress_ratio, quantize_param.numel * 32)
                quantize_param_bit.update(quantize_param.param_quantize_bit, quantize_param.param_nonzero_num)
                quantize_index_bit.update(quantize_param.index_quantize_bit, quantize_param.index_chunk_num)
                quantize_param_codebook_data_size.update(quantize_param.param_codebook_data_size)
                quantize_param_data_size.update(quantize_param.param_data_size)
                quantize_index_data_size.update(quantize_param.index_data_size)
                quantize_data_size.update(quantize_param.data_size)
                quantize_overall_data_size.update(quantize_param.data_size)
                quantize_overall_param_data_size.update(quantize_param.param_data_size)
            quantize_overall_data_size.update(param.data.numel()*32)
            quantize_overall_param_data_size.update(param.data.numel() * 32)
            quantize_overall_compress_ratio.update(1, param.data.numel() * 32)
        print('Quantize Param Bit | {}'.format(quantize_param_bit.avg))
        print('Quantize Index Bit | {}'.format(quantize_index_bit.avg))
        print('Quantize Compress Ratio | {}'.format(quantize_compress_ratio.avg))
        print('Quantize Overall Compress Ratio | {}'.format(quantize_overall_compress_ratio.avg))
        print('Quantize Param Codebook Size | {}'.format(quantize_param_codebook_data_size.sum))
        print('Quantize Param Data Size | {}'.format(quantize_param_data_size.sum))
        print('Quantize Index Data Size | {}'.format(quantize_index_data_size.sum))
        print('Quantize Data Size | {}'.format(quantize_data_size.sum))
        print('Quantize Overall Param Data Size | {}'.format(quantize_overall_param_data_size.sum))
        print('Quantize Overall Data Size | {}'.format(quantize_overall_data_size.sum))
        print_section('Stop Quantize Encoding')
        return replica, quantize_parameters

    @staticmethod
    def decode(network, encode_parameters):
        assert isinstance(network, torch.nn.Module)
        print_section('Start Quantize Decoding')

        def _worker(i, name, param, quantize_parameter):
            if name == quantize_parameter[0]:
                print('Decoding ' + name)
                quantize_param = quantize_parameter[1]
                param.data = param.data.new(quantize_param.shape).copy_(quantize_param.data)

        for i, (name, param) in enumerate(network.named_parameters()):
            if i in encode_parameters:
                _worker(i, name, param, encode_parameters[i])
        print_section('Stop Quantize Decoding')
