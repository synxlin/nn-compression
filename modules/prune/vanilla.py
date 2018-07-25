import math
import torch


def prune_vanilla_elementwise(param, sparsity):
    """
    element-wise vanilla pruning
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: pruning sparsity
    :return: torch.(cuda.)ByteTensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    num_param = param.numel()
    param_abs = param.abs()
    num_pruned = int(math.ceil(num_param * sparsity))
    num_stayed = num_param - num_pruned
    if sparsity <= 0.5:
        _, topk_indices = torch.topk(param_abs.view(num_param), num_pruned,
                                     0, largest=False, sorted=False)
        mask = torch.zeros_like(param).byte()
        param.view(num_param).index_fill_(0, topk_indices, 0)
        mask.view(num_param).index_fill_(0, topk_indices, 1)
    else:
        thr = torch.min(torch.topk(param_abs.view(num_param), num_stayed,
                                   0, largest=True, sorted=False)[0])
        mask = torch.lt(param_abs, thr)
        param.masked_fill_(mask, 0)
    return mask


def prune_vanilla_kernelwise(param, sparsity):
    """
    kernel-wise vanilla pruning, the importance determined by L1 norm
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: pruning sparsity
    :return: torch.(cuda.)ByteTensor, mask for zeros
    """
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    num_kernels = param.size(0) * param.size(1)
    param_k = param.view(num_kernels, -1)
    param_abs = param_k.abs().sum(1)  # L1-norm importance
    num_pruned = int(math.ceil(num_kernels * sparsity))
    _, topk_indices = torch.topk(param_abs, num_pruned,
                                 0, largest=False, sorted=False)
    mask = torch.zeros_like(param).byte()
    mask_k = mask.view(num_kernels, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask


def prune_vanilla_filterwise(param, sparsity):
    """
    filter-wise vanilla pruning, the importance determined by L1 norm
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: pruning sparsity
    :return: torch.(cuda.)ByteTensor, mask for zeros
    """
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    num_filters = param.size(0)
    param_k = param.view(num_filters, -1)
    param_abs = param_k.abs().sum(1)  # L1-norm importance
    num_pruned = int(math.ceil(num_filters * sparsity))
    _, topk_indices = torch.topk(param_abs, num_pruned,
                                 0, largest=False, sorted=False)
    mask = torch.zeros_like(param).byte()
    mask_k = mask.view(num_filters, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask