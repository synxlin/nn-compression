import math
import torch

from slender.prune.vanilla import prune_vanilla_elementwise, prune_vanilla_kernelwise, \
    prune_vanilla_filterwise, VanillaPruner


def test_prune_vanilla_elementwise():
    param = torch.rand(64, 128, 3, 3)
    mask = prune_vanilla_elementwise(sparsity=0.3, param=param)
    assert mask.sum() == int(math.ceil(param.numel() * 0.3))
    assert param.masked_select(mask).eq(0).all()
    mask = prune_vanilla_elementwise(sparsity=0.7, param=param)
    assert mask.sum() == int(math.ceil(param.numel() * 0.7))
    assert param.masked_select(mask).eq(0).all()


def test_prune_vanilla_kernelwise():
    param = torch.rand(64, 128, 3, 3)
    mask = prune_vanilla_kernelwise(sparsity=0.5, param=param)
    mask_s = mask.view(64*128, -1).all(1).sum()
    assert mask_s == 32*128
    assert param.masked_select(mask).eq(0).all()


def test_prune_vanilla_filterwise():
    param = torch.rand(64, 128, 3, 3)
    mask = prune_vanilla_filterwise(sparsity=0.5, param=param)
    mask_s = mask.view(64, -1).all(1).sum()
    assert mask_s == 32
    assert param.masked_select(mask).eq(0).all()


def test_vanilla_pruner():
    rule = [
        ('0.weight', 'element', [0.3, 0.5]),
        ('1.weight', 'element', [0.4, 0.6])
    ]
    rule_dict = {
        '0.weight': [0.3, 0.5],
        '1.weight': [0.4, 0.6]
    }
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    pruner = VanillaPruner(rule=rule)
    pruner.prune(model=model, stage=0, verbose=True)
    for n, param in model.named_parameters():
        if param.dim() > 1:
            mask = pruner.masks[n]
            assert mask.sum() == int(math.ceil(param.numel() * rule_dict[n][0]))
            assert param.data.masked_select(mask).eq(0).all()
    state_dict = pruner.state_dict()
    pruner = VanillaPruner().load_state_dict(state_dict)
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    pruner.prune(model=model, stage=0)
    for n, param in model.named_parameters():
        if param.dim() > 1:
            mask = pruner.masks[n]
            assert mask.sum() == int(math.ceil(param.numel() * rule_dict[n][0]))
            assert param.data.masked_select(mask).eq(0).all()
    pruner.prune(model=model, stage=1, update_masks=True, verbose=True)
    for n, param in model.named_parameters():
        if param.dim() > 1:
            mask = pruner.masks[n]
            assert mask.sum() == int(math.ceil(param.numel() * rule_dict[n][1]))
            assert param.data.masked_select(mask).eq(0).all()
