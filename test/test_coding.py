import torch

from modules.prune.vanilla import prune_vanilla_elementwise
from modules.quantize.linear import quantize_linear_fix_zeros
from modules.quantize.fixed_point import quantize_fixed_point
from modules.quantize.quantizer import Quantizer
from modules.coding.encode import EncodedParam
from modules.coding.codec import Codec


def test_encode_param():
    param = torch.rand(256, 128, 3, 3)
    prune_vanilla_elementwise(sparsity=0.7, param=param)
    quantize_linear_fix_zeros(param, k=16)
    huffman = EncodedParam(param=param, method='huffman',
                           encode_indices=True, bit_length_zero_run_length=4)
    stats = huffman.stats
    print(stats)
    assert torch.eq(param, huffman.data).all()
    state_dict = huffman.state_dict()
    huffman = EncodedParam()
    huffman.load_state_dict(state_dict)
    assert torch.eq(param, huffman.data).all()
    vanilla = EncodedParam(param=param, method='vanilla',
                           encode_indices=True, bit_length_zero_run_length=4)
    stats = vanilla.stats
    print(stats)
    assert torch.eq(param, vanilla.data).all()
    quantize_fixed_point(param=param, bit_length=4, bit_length_integer=0)
    fixed_point = EncodedParam(param=param, method='fixed_point',
                               bit_length=4, bit_length_integer=0,
                               encode_indices=True, bit_length_zero_run_length=4)
    stats = fixed_point.stats
    print(stats)
    assert torch.eq(param, fixed_point.data).all()


def test_codec():
    quantize_rule = [
        ('0.weight', 'k-means', 4, 'k-means++'),
        ('1.weight', 'fixed_point', 6, 1),
    ]
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    mask_dict = {}
    for n, p in model.named_parameters():
        mask_dict[n] = prune_vanilla_elementwise(sparsity=0.6, param=p.data)
    quantizer = Quantizer(rule=quantize_rule, fix_zeros=True)
    quantizer.quantize(model, update_labels=False, verbose=True)
    rule = [
        ('0.weight', 'huffman', 0, 0, 4),
        ('1.weight', 'fixed_point', 6, 1, 4)
    ]
    codec = Codec(rule=rule)
    encoded_module = codec.encode(model)
    print(codec.stats)
    state_dict = encoded_module.state_dict()
    model_2 = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                  torch.nn.Conv2d(128, 512, 1, bias=False))
    model_2 = Codec.decode(model_2, state_dict)
    for p1, p2 in zip(model.parameters(), model_2.parameters()):
        if p1.dim() > 1:
            assert torch.eq(p1, p2).all()
