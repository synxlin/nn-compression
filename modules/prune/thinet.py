import math
import random

import torch


def channel_selection_thinet(sparsity, output_feature, fn_next_output_feature, method='greedy'):
    """
    select channel to prune with a given metric
    :param sparsity: float, pruning sparsity
    :param output_feature: torch.(cuda.)Tensor, output feature map of the layer being pruned
    :param fn_next_output_feature: function, function to calculate the next output feature map
    :param method: str
        'greedy': select one contributed to the smallest next feature after another
        'random': randomly select
    :return:
        list of int, indices of filters to be pruned
    """
    num_channel = output_feature.size(1)
    num_pruned = int(math.floor(num_channel * sparsity))

    if method == 'greedy':
        indices_pruned = []
        while len(indices_pruned) < num_pruned:
            min_diff = 1e10
            min_idx = 0
            for idx in range(num_channel):
                if idx in indices_pruned:
                    continue
                indices_try = indices_pruned + [idx]
                output_feature_try = torch.zeros_like(output_feature)
                output_feature_try[:, indices_try, ...] = output_feature[:, indices_try, ...]
                output_feature_try = fn_next_output_feature(output_feature_try)
                output_feature_try_norm = output_feature_try.norm(2)
                if output_feature_try_norm < min_diff:
                    min_diff = output_feature_try_norm
                    min_idx = idx
            indices_pruned.append(min_idx)
    elif method == 'random':
        indices_pruned = random.sample(range(num_channel), num_pruned)
    else:
        raise NotImplementedError

    return indices_pruned


def module_surgery_thinet(module, next_module, indices_pruned):
    """
    prune the redundant filters/channels
    :param module: torch.nn.module, module of the layer being pruned
    :param next_module: torch.nn.module, module of the next layer to the one being pruned
    :param indices_pruned: list of int, indices of filters/channels to be pruned
    :return:
        void
    """
    # operate module
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        indices_stayed = list(set(range(module.out_channels)) - set(indices_pruned))
        num_channels_stayed = len(indices_stayed)
        module.out_channels = num_channels_stayed
    elif isinstance(module, torch.nn.Linear):
        indices_stayed = list(set(range(module.out_features)) - set(indices_pruned))
        num_channels_stayed = len(indices_stayed)
        module.out_features = num_channels_stayed
    else:
        raise NotImplementedError
    # operate module weight
    new_weight = module.weight[indices_stayed, ...].clone()
    del module.weight
    module.weight = torch.nn.Parameter(new_weight)
    # operate module bias
    if module.bias is not None:
        new_bias = module.bias[indices_stayed, ...].clone()
        del module.bias
        module.bias = torch.nn.Parameter(new_bias)
    # operate next_module
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        next_module.in_channels = num_channels_stayed
    elif isinstance(next_module, torch.nn.Linear):
        next_module.in_features = num_channels_stayed
    else:
        raise NotImplementedError
    # operate next_module weight
    new_weight = next_module.weight[:, indices_stayed, ...].clone()
    del next_module.weight
    next_module.weight = torch.nn.Parameter(new_weight)


def weight_reconstruction_thinet(next_module, next_input_feature, next_output_feature, cpu=True):
    """
    reconstruct the weight of the next layer to the one being pruned
    :param next_module: torch.nn.module, module of the next layer to the one being pruned
    :param next_input_feature: torch.(cuda.)Tensor, new input feature map of the next layer
    :param next_output_feature: torch.(cuda.)Tensor, original output feature map of the next layer
    :param cpu: bool, whether done in cpu
    :return:
        void
    """
    if next_module.bias is not None:
        bias_size = [1] * next_output_feature.dim()
        bias_size[1] = -1
        next_output_feature -= next_module.bias.view(bias_size)
    if cpu:
        next_input_feature = next_input_feature.cpu()
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        unfold = torch.nn.Unfold(kernel_size=next_module.kernel_size,
                                 dilation=next_module.dilation,
                                 padding=next_module.padding,
                                 stride=next_module.stride)
        if not cpu:
            unfold = unfold.cuda()
        next_input_feature = unfold(next_input_feature)
        next_input_feature = next_input_feature.transpose(1, 2)
        num_fields = next_input_feature.size(0) * next_input_feature.size(1)
        next_input_feature = next_input_feature.reshape(num_fields, -1)
        next_output_feature = next_output_feature.view(next_output_feature.size(0), next_output_feature.size(1), -1)
        next_output_feature = next_output_feature.transpose(1, 2).reshape(num_fields, -1)
    if cpu:
        next_output_feature = next_output_feature.cpu()
    param, _ = torch.gels(next_output_feature, next_input_feature)
    param = param[0:next_input_feature.size(1), :].clone().t().contiguous().view(next_output_feature.size(1), -1)
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        param = param.view(next_module.out_channels, next_module.in_channels, *next_module.kernel_size)
    next_module.weight = torch.nn.Parameter(param)


def prune_thinet(sparsity, module, next_module, input_feature, fn_next_input_feature, method='greedy', cpu=True):
    """
    ThiNet pruning core function
    :param sparsity: float, pruning sparsity
    :param module: torch.nn.module, module of the layer being pruned
    :param next_module: torch.nn.module, module of the next layer to the one being pruned
    :param input_feature: torch.(cuda.)Tensor, input feature map of the layer being pruned
    :param fn_next_input_feature: function, function to calculate the input feature map for next_module
    :param method: str
        'greedy': select one contributed to the smallest next feature after another
        'random': randomly select
    :param cpu: bool, whether done in cpu for larger reconstruction batch size
    :return:
        void
    """
    assert input_feature.dim() >= 2  # N x C x ...
    output_feature = module(input_feature)
    next_input_feature = fn_next_input_feature(output_feature)
    next_output_feature = next_module(next_input_feature)

    def fn_next_output_feature(feature):
        return next_module(fn_next_input_feature(feature))

    indices_pruned = channel_selection_thinet(sparsity=sparsity, output_feature=output_feature,
                                              fn_next_output_feature=fn_next_output_feature, method=method)
    module_surgery_thinet(module=module, next_module=next_module, indices_pruned=indices_pruned)

    next_input_feature = fn_next_input_feature(module(input_feature))
    weight_reconstruction_thinet(next_module=next_module, next_input_feature=next_input_feature,
                                 next_output_feature=next_output_feature, cpu=cpu)
