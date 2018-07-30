import torch

from modules.quantize.linear import quantize_linear, quantize_linear_fix_zeros
from modules.quantize.kmeans import quantize_k_means, quantize_k_means_fix_zeros
from modules.quantize.quantizer import Quantizer


def test_quantize_linear():
    param = torch.rand(128, 64, 3, 3)
    codebook = quantize_linear(param, k=16)
    assert codebook['centers_'].numel() == 16
    centers_ = codebook['centers_'].tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_


def test_quantize_linear_fix_zeros():
    param = torch.rand(128, 64, 3, 3)
    from modules.prune.vanilla import prune_vanilla_elementwise
    mask = prune_vanilla_elementwise(sparsity=0.4, param=param)
    codebook = quantize_linear_fix_zeros(param, k=16)
    assert codebook['centers_'].numel() == 16
    centers_ = codebook['centers_'].tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()


def test_quantize_k_means():
    param = torch.rand(128, 64, 3, 3)
    codebook = quantize_k_means(param)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    param = torch.rand(128, 64, 3, 3)
    codebook = quantize_k_means(param, codebook, update_labels=True,
                                update_centers=True)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_


def test_quantize_k_means_fix_zeros():
    param = torch.rand(128, 64, 3, 3)
    from modules.prune.vanilla import prune_vanilla_elementwise
    mask = prune_vanilla_elementwise(sparsity=0.4, param=param)
    codebook = quantize_k_means_fix_zeros(param)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()
    codebook = quantize_k_means_fix_zeros(param, codebook, update_labels=True,
                                          update_centers=True)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()


def test_quantizer():
    rule = [
        ('0.weight', 4, 'k-means', 'k-means++'),
        ('1.weight', 6, 'k-means', 'k-means++'),
    ]
    rule_dict = {
        '0.weight': 4,
        '1.weight': 6
    }
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    quantizer = Quantizer(rule=rule, fix_zeros=True)