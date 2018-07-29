import re
import torch

from .encode import EncodedParam, EncodedModule
from ..utils import AverageMeter


class Codec(object):
    def __init__(self, rule):
        """
        Codec for coding
        :param rule: str, path to the rule file, each line formats
                        'param_name coding_method bit_length_fixed_point bit_length_fixed_point_of_integer_part
                         bit_length_of_zero_run_length'
                     list of tuple,
                        [(param_name(str), coding_method(str), bit_length_fixed_point(int),
                         bit_length_fixed_point_of_integer_part(int), bit_length_of_zero_run_length(int))]
        """
        if isinstance(rule, str):
            content = map(lambda x: x.split(), open(rule).readlines())
            content = filter(lambda x: len(x) == 5, content)
            rule = list(map(lambda x: (x[0], x[1], int(x[2]), int(x[3]), int(x[4])), content))
        assert isinstance(rule, list) or isinstance(rule, tuple)
        self.rule = rule
        self.stats = {
            'compression_ratio': {
                'compressed': AverageMeter(),
                'total': AverageMeter()
            },
            'memory_size': {
                'codebook': AverageMeter(),
                'param': AverageMeter(),
                'compressed_param': AverageMeter(),
                'index': AverageMeter(),
                'total': AverageMeter()
            },
            'detail': dict()
        }

        print("=" * 89)
        print("Initializing Huffman Codec\n"
              "Rules\n"
              "{rule}".format(rule=self.rule))
        print("=" * 89)

    def reset_stats(self):
        """
        reset stats of codec
        :return:
            void
        """
        self.stats['detail'] = dict()
        for _, v in self.stats['compression_ratio'].items():
            v.reset()
        for _, v in self.stats['memory_size'].items():
            v.reset()

    def encode_param(self, param, param_name):
        """
        encode the parameters based on rule
        :param param: torch.(cuda.)tensor, parameter
        :param param_name: str, name of parameter
        :return:
            EncodedParam, encoded parameter
        """
        rule_id = -1
        for idx, x in enumerate(self.rule):
            m = re.match(x[0], param_name)
            if m is not None and len(param_name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            rule = self.rule[rule_id]
            encoded_param = EncodedParam(param, method=rule[1],
                                         bit_length=rule[2], bit_length_integer=rule[3],
                                         encode_indices=True, bit_length_zero_run_length=rule[4])
            return encoded_param
        else:
            return None

    def encode(self, model):
        """
        encode network based on rule
        :param model: torch.(cuda.)module, network model
        :return:
            EncodedModule, encoded model
        """
        assert isinstance(model, torch.nn.Module)
        self.reset_stats()
        encoded_params = dict()
        print("=" * 89)
        print("Start Encoding")
        print("=" * 89)
        print("{:^30} | {:<25} | {:<25} | {:<25} | {:<25} | {:<25} | {:<25} | {:<25}".
              format('Param Name', 'Param Density', 'Param Bit', 'Index Bit', 'Param Mem',
                     'Index Mem', 'Codebook Mem', 'Compression Ratio'))
        for param_name, param in model.named_parameters():
            if 'AuxLogits' in param_name:
                # deal with googlenet
                continue
            encoded_param = self.encode_param(param=param.data, param_name=param_name)
            if encoded_param is not None:
                # check encoded result
                assert torch.equal(param.data, encoded_param.data)
                stats = encoded_param.stats
                print("{param_name:^30} | {density:<25} | {bit_param:<25} | {bit_index:<25} | "
                      "{mem_param:<25} | {mem_index:<25} | {mem_codebook:<25} | {compression_ratio:<25}"
                      .format(param_name=param_name, density=stats['num_nz'] / stats['num_el'],
                              bit_param=stats['bit_length']['param'], bit_index=stats['bit_length']['index'],
                              mem_param=stats['memory_size']['param'], mem_index=stats['memory_size']['index'],
                              mem_codebook=stats['memory_size']['codebook'],
                              compression_ratio=stats['compression_ratio']))
                encoded_params[param_name] = encoded_param.state_dict()
                # statistics
                self.stats['compression_ratio']['compressed'].accumulate(stats['memory_size']['total'],
                                                                         stats['num_el'] * 32)
                self.stats['compression_ratio']['total'].accumulate(stats['memory_size']['total'],
                                                                    stats['num_el'] * 32)
                self.stats['memory_size']['codebook'].accumulate(stats['memory_size']['codebook'])
                self.stats['memory_size']['param'].accumulate(stats['memory_size']['param'])
                self.stats['memory_size']['index'].accumulate(stats['memory_size']['index'])
                self.stats['memory_size']['compressed_param'].accumulate(stats['memory_size']['param'])
                self.stats['detail'][param_name] = stats
            else:
                print("{:<30} | skipping".format(param_name))
                memory_size_param = param.data.numel() * 32
                self.stats['compression_ratio']['total'].accumulate(memory_size_param, memory_size_param)
                self.stats['memory_size']['param'].accumulate(memory_size_param)
        print("=" * 89)
        print("Stop Encoding")
        print("=" * 89)
        print("Compress Ratio               | {}\n"
              "Overall Compress Ratio       | {}\n"
              "Codebook Memory Size         | {}\n"
              "Compressed Param Memory Size | {}\n"
              "Index Memory Size            | {}\n"
              "Overall Param Memory Size    | {}"
              .format(self.stats['compression_ratio']['compressed'].avg,
                      self.stats['compression_ratio']['total'].avg,
                      self.stats['memory_size']['codebook'].sum,
                      self.stats['memory_size']['compressed_param'].sum,
                      self.stats['memory_size']['index'].sum,
                      self.stats['memory_size']['param'].sum))
        print("=" * 89)
        return EncodedModule(module=model, encoded_param=encoded_params)

    @staticmethod
    def decode(model, state_dict):
        """
        decode the network using state dict from EncodedModule
        :param model: torch.nn.module, network model
        :param state_dict: state dict from EncodedModule
        :return:
            torch.nn.module, decoded network
        """
        assert isinstance(model, torch.nn.Module)
        print("=" * 89)
        print("Start Decoding")
        for param_name, _ in model.named_parameters():
            if param_name in state_dict:
                print("Decoding {}".format(param_name))
                state_dict[param_name] = state_dict[param_name].data
        model = model.load_state_dict(state_dict)
        print("Stop Decoding")
        print("=" * 89)
        return model
