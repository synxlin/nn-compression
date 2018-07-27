import math
from bitarray import bitarray


def get_vanilla_codebook(symb):
    codebook = dict()
    symb = set(symb)
    bit_length = math.log(len(symb), 2)
    bit_format = '{:0%db}' % bit_length
    for i, s in enumerate(symb):
        codebook[s] = bitarray(bit_format.format(i))
    return codebook
