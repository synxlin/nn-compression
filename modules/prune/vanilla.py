import re
import math
import torch


def prune_vanilla_elementwise(sparsity, param):
    """
    element-wise vanilla pruning
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
    :return:
        torch.(cuda.)ByteTensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).byte()
    num_el = param.numel()
    param_abs = param.abs()
    num_pruned = int(math.ceil(num_el * sparsity))
    num_stayed = num_el - num_pruned
    if sparsity <= 0.5:
        _, topk_indices = torch.topk(param_abs.view(num_el), num_pruned,
                                     0, largest=False, sorted=False)
        mask = torch.zeros_like(param).byte()
        param.view(num_el).index_fill_(0, topk_indices, 0)
        mask.view(num_el).index_fill_(0, topk_indices, 1)
    else:
        thr = torch.min(torch.topk(param_abs.view(num_el), num_stayed,
                                   0, largest=True, sorted=False)[0])
        mask = torch.lt(param_abs, thr)
        param.masked_fill_(mask, 0)
    return mask


def prune_vanilla_kernelwise(sparsity, param):
    """
    kernel-wise vanilla pruning, the importance determined by L1 norm
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
    :return:
        torch.(cuda.)ByteTensor, mask for zeros
    """
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).byte()
    num_kernels = param.size(0) * param.size(1)
    param_k = param.view(num_kernels, -1)
    param_norm = param_k.norm(1, -1)  # L1-norm importance
    num_pruned = int(math.ceil(num_kernels * sparsity))
    _, topk_indices = torch.topk(param_norm, num_pruned,
                                 0, largest=False, sorted=False)
    mask = torch.zeros_like(param).byte()
    mask_k = mask.view(num_kernels, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask


def prune_vanilla_filterwise(sparsity, param):
    """
    filter-wise vanilla pruning, the importance determined by L1 norm
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
    :return:
        torch.(cuda.)ByteTensor, mask for zeros
    """
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).byte()
    num_filters = param.size(0)
    param_k = param.view(num_filters, -1)
    param_norm = param_k.norm(1, -1)  # L1-norm importance
    num_pruned = int(math.ceil(num_filters * sparsity))
    _, topk_indices = torch.topk(param_norm, num_pruned,
                                 0, largest=False, sorted=False)
    mask = torch.zeros_like(param).byte()
    mask_k = mask.view(num_filters, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask


class VanillaPruner(object):
    """

    """
    def __init__(self, rule, granularity='element'):
        """

        :param rule:
        :param granularity:
        """
        if isinstance(rule, str):
            content = map(lambda x: x.split(), open(rule).readlines())
            content = filter(lambda x: len(x) == 2, content)
            rule = list(map(lambda x: (x[0], list(map(float, x[1].split(',')))), content))
        assert isinstance(rule, list)
        self.rule = rule

        if granularity == 'element':
            self.prune = prune_vanilla_elementwise
        elif granularity == 'kernel':
            self.prune = prune_vanilla_kernelwise
        elif granularity == 'filter':
            self.prune = prune_vanilla_filterwise
        else:
            raise NotImplementedError
        self.granularity = granularity

        self.masks = dict()

        print("Initializing Vanilla Pruner\nRules:\n{}".format(self.rule))

    def load_state_dict(self, state_dict):
        """

        :param state_dict:
        :return:
        """
        self.rule = state_dict['rule']
        self.granularity = state_dict['granularity']
        self.masks = state_dict['masks']
        print("Customizing Vanilla Pruner\nRules:\n{}".format(self.rule))

    def state_dict(self):
        """

        :return:
        """
        state_dict = dict()
        state_dict['rule'] = self.rule
        state_dict['granularity'] = self.granularity
        state_dict['masks'] = self.masks

    def prune_param(self, param, param_name, stage=0):
        """

        :param param:
        :param param_name:
        :param stage:
        :return:
        """
        rule_id = -1
        for idx, x in enumerate(self.rule):
            m = re.match(x[0], param_name)
            if m is not None and len(param_name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            max_num_stage = len(self.rule[rule_id][1]) - 1
            stage = min(max(0, stage), max_num_stage)
            sparsity = self.rule[rule_id][1][stage]
            print("{}:\t\tstage: {}\t\tsparsity: {.3f}".format(param_name, stage, sparsity))
            mask = self.prune(sparsity=sparsity, param=param)
            return mask
        else:
            print("{}:\t\tskipping".format(param_name))
            return None

    def prune(self, model, stage=0, update_masks=False):
        """

        :param model:
        :param stage:
        :param update_masks:
        :return:
        """
        update_masks = True if update_masks or len(self.masks) == 0 else False
        if update_masks:
            print("------ updating masks ------")
        for idx, (param_name, param) in enumerate(model.named_parameters()):
            if param.dim() > 1:
                if update_masks:
                    mask = self.prune_param(param=param, param_name=param_name, stage=stage)
                    if mask is not None:
                        self.masks[param_name] = mask
                else:
                    if param_name in self.masks:
                        mask = self.masks[param_name]
                        param.masked_fill_(mask, 0)
