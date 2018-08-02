import torch

from modules.prune.vanilla import prune_vanilla_elementwise
from modules.quantize.linear import quantize_linear, quantize_linear_fix_zeros
from modules.quantize.kmeans import quantize_k_means, quantize_k_means_fix_zeros
from modules.quantize.fixed_point import quantize_fixed_point
from modules.quantize.quantizer import VanillaQuantizer


def test_quantize_linear():
    param = torch.rand(128, 64, 3, 3) - 0.5
    codebook = quantize_linear(param, k=16)
    assert codebook['cluster_centers_'].numel() == 16
    centers_ = codebook['cluster_centers_'].tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_


def test_quantize_linear_fix_zeros():
    param = torch.rand(128, 64, 3, 3) - 0.5
    mask = prune_vanilla_elementwise(sparsity=0.4, param=param)
    codebook = quantize_linear_fix_zeros(param, k=16)
    assert codebook['cluster_centers_'].numel() == 16
    centers_ = codebook['cluster_centers_'].tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()


def test_quantize_k_means():
    param = torch.rand(128, 64, 3, 3) - 0.5
    codebook = quantize_k_means(param, k=16)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    param = torch.rand(128, 64, 3, 3)
    codebook = quantize_k_means(param, k=16, codebook=codebook,
                                update_centers=True)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_


def test_quantize_k_means_fix_zeros():
    param = torch.rand(128, 64, 3, 3) - 0.5
    mask = prune_vanilla_elementwise(sparsity=0.4, param=param)
    codebook = quantize_k_means_fix_zeros(param, k=16)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()
    codebook = quantize_k_means_fix_zeros(param, k=16, codebook=codebook,
                                          update_centers=True)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()


def test_quantized_fixed_point():
    param = torch.rand(128, 64, 3, 3) - 0.5
    mask = prune_vanilla_elementwise(sparsity=0.4, param=param)
    codebook = quantize_fixed_point(param, bit_length=8, bit_length_integer=1)
    assert codebook['cluster_centers_'].numel() == 2 ** 8
    centers_ = codebook['cluster_centers_'].tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()


def test_quantizer():
    rule = [
        ('0.weight', 'k-means', 4, 'k-means++'),
        ('1.weight', 'fixed_point', 6, 1),
    ]
    rule_dict = {
        '0.weight': ['k-means', 16],
        '1.weight': ['fixed_point', 6, 1]
    }
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    mask_dict = {}
    for n, p in model.named_parameters():
        mask_dict[n] = prune_vanilla_elementwise(sparsity=0.4, param=p)
    quantizer = VanillaQuantizer(rule=rule, fix_zeros=True)
    quantizer.quantize(model, update_labels=False, verbose=True)
    for n, p in model.named_parameters():
        if n in rule_dict:
            vals = set(p.data.view(p.numel()).tolist())
            if rule_dict[n][0] == 'k-means':
                centers_ = quantizer.codebooks[n].cluster_centers_.view(rule_dict[n][1]).tolist()
            else:
                centers_ = quantizer.codebooks[n]['cluster_centers_']
            for v in vals:
                assert v in centers_
            assert p.data.masked_select(mask_dict[n]).eq(0).all

    state_dict = quantizer.state_dict()
    quantizer = Quantizer().load_state_dict(state_dict)
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    mask_dict = {}
    for n, p in model.named_parameters():
        mask_dict[n] = prune_vanilla_elementwise(sparsity=0.4, param=p)
    quantizer.quantize(model, update_labels=True, verbose=True)
    for n, p in model.named_parameters():
        if n in rule_dict:
            vals = set(p.data.view(p.numel()).tolist())
            if rule_dict[n][0] == 'k-means':
                centers_ = quantizer.codebooks[n].cluster_centers_.view(rule_dict[n][1]).tolist()
            else:
                centers_ = quantizer.codebooks[n]['cluster_centers_']
            for v in vals:
                assert v in centers_
            assert p.data.masked_select(mask_dict[n]).eq(0).all