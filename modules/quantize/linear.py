import math
import torch

magic_percentile = 0.001


# TODO: fixed point arg
def quantize_linear(param, k=16, **unused):
    """

    :param param:
    :param k:
    :param unused:
    :return:
    """
    num_el = param.numel()
    kth = int(math.ceil(num_el * magic_percentile))
    param_flatten = param.view(num_el)
    param_min, _ = torch.topk(param_flatten, kth, dim=0, largest=False, sorted=False)
    param_min = param_min.max()
    param_max, _ = torch.topk(param_flatten, kth, dim=0, largest=True, sorted=False)
    param_max = param_max.min()
    step = (param_max - param_min) / (k - 1)
    param.clamp_(param_min, param_max).sub_(param_min).div_(step).round_().mul_(step).add_(param_min)
    codebook = {'centers_': torch.tensor(list(set(param_flatten.cpu().tolist())))}
    # codebook = torch.linspace(param_min, param_max, k)
    return codebook


# TODO: fixed point arg
def quantize_linear_fix_zeros(param, k=16, **unused):
    """

    :param param:
    :param k:
    :param unused:
    :return:
    """
    zero_mask = torch.eq(param, 0.0)  # get zero mask
    num_param = param.numel()
    kth = int(math.ceil(num_param * magic_percentile))
    param_flatten = param.view(num_param)
    param_min, _ = torch.topk(param_flatten, kth, dim=0, largest=False, sorted=False)
    param_min = param_min.max()
    param_max, _ = torch.topk(param_flatten, kth, dim=0, largest=True, sorted=False)
    param_max = param_max.min()
    step = (param_max - param_min) / (k - 2)
    param.clamp_(param_min, param_max).sub_(param_min).div_(step).round_().mul_(step).add_(param_min)
    param.masked_fill_(zero_mask, 0)  # recover zeros
    codebook = {'centers_': torch.tensor(list(set(param_flatten.cpu().tolist())))}
    # codebook = torch.linspace(param_min, param_max, k - 1)
    return codebook