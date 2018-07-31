import torch


def quantize_fixed_point(param, bit_length=8, bit_length_integer=0, **unused):
    """
    vanilla fixed point quantization, inherently fixing zeros
    :param param: torch.(cuda.)tensor
    :param bit_length: int, bit length of fixed point param
                        including sign bit, default=8
    :param bit_length_integer: int, bit length of integer part
                                    of fixed point param, default=0
    :param unused: unused: unused options
    :return:
    """
    mul_coeff = 2 ** (bit_length - 1 - bit_length_integer)
    div_coeff = 2 ** (bit_length_integer - bit_length + 1)
    max_coeff = 2 ** (bit_length - 1)
    param.mul_(mul_coeff).floor_().clamp_(-max_coeff - 1, max_coeff - 1).mul_(div_coeff)
    codebook = {'cluster_centers_': torch.arange(-max_coeff * div_coeff,
                                                 max_coeff * div_coeff, div_coeff),
                'method': 'fixed_point',
                }
    return codebook