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
        _, topk_indices = torch.topk(param_abs.view(num_el), k=num_pruned,
                                     dim=0, largest=False, sorted=False)
        mask = torch.zeros_like(param).byte()
        param.view(num_el).index_fill_(0, topk_indices, 0)
        mask.view(num_el).index_fill_(0, topk_indices, 1)
    else:
        thr = torch.min(torch.topk(param_abs.view(num_el), k=num_stayed,
                                   dim=0, largest=True, sorted=False)[0])
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
    _, topk_indices = torch.topk(param_norm, k=num_pruned,
                                 dim=0, largest=False, sorted=False)
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
    _, topk_indices = torch.topk(param_norm, k=num_pruned,
                                 dim=0, largest=False, sorted=False)
    mask = torch.zeros_like(param).byte()
    mask_k = mask.view(num_filters, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask


class VanillaPruner(object):

    def __init__(self, rule=None, granularity='element'):
        """
        Pruner Class for Vanilla Pruning Method
        :param rule: str, path to the rule file, each line formats 'param_name sparsity_stage_0, sparstiy_stage_1, ...'
                     list of tuple, [(param_name(str), [sparsity_stage_0, sparsity_stage_1, ...])]
        :param granularity: str, pruning granularity, choose from ['element', 'kernel', 'filter']
        """
        if isinstance(rule, str):
            content = map(lambda x: x.split(), open(rule).readlines())
            content = filter(lambda x: len(x) == 2, content)
            rule = list(map(lambda x: (x[0], list(map(float, x[1].split(',')))), content))
        assert isinstance(rule, list) or isinstance(rule, tuple) or rule is None
        self.rule = rule
        self.max_num_stage = 0 if rule is None else max(map(lambda x: len(x[1]), rule))

        if granularity == 'element':
            self.fn_prune = prune_vanilla_elementwise
        elif granularity == 'kernel':
            self.fn_prune = prune_vanilla_kernelwise
        elif granularity == 'filter':
            self.fn_prune = prune_vanilla_filterwise
        else:
            raise NotImplementedError
        self.granularity = granularity

        self.masks = dict()

        print("=" * 89)
        if self.rule is None:
            print("Initializing Vanilla Pruner WITHOUT rules")
        else:
            print("Initializing Vanilla Pruner with rules:")
            for r in self.rule:
                print(r)
        print("=" * 89)

    def load_state_dict(self, state_dict, keep_rule=False):
        """
        Recover Pruner
        :param state_dict: dict, a dictionary containing a whole state of the Pruner
        :param keep_rule: bool, whether to keep rule and granularity settings
        :return: VanillaPruner
        """
        if not keep_rule:
            self.rule = state_dict['rule']
            self.max_num_stage = max(map(lambda x: len(x[1]), self.rule))
            self.granularity = granularity = state_dict['granularity']
            if granularity == 'element':
                self.fn_prune = prune_vanilla_elementwise
            elif granularity == 'kernel':
                self.fn_prune = prune_vanilla_kernelwise
            elif granularity == 'filter':
                self.fn_prune = prune_vanilla_filterwise
            else:
                raise NotImplementedError
        self.masks = state_dict['masks']
        print("=" * 89)
        print("Customizing Vanilla Pruner with rules:")
        for r in self.rule:
            print(r)
        print("=" * 89)

    def state_dict(self):
        """
        Returns a dictionary containing a whole state of the Pruner
        :return: dict, a dictionary containing a whole state of the Pruner
        """
        state_dict = dict()
        state_dict['rule'] = self.rule
        state_dict['granularity'] = self.granularity
        state_dict['masks'] = self.masks
        return state_dict

    def prune_param(self, param, param_name, stage=0, verbose=False):
        """
        prune parameter
        :param param: torch.(cuda.)tensor
        :param param_name: str, name of param
        :param stage: int, the pruning stage, default=0
        :param verbose: bool, whether to print the pruning details
        :return:
            torch.(cuda.)ByteTensor, mask for zeros
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
            if verbose:
                print("{param_name:^30} | {stage:5d} | {spars:.3f}".
                      format(param_name=param_name, stage=stage, spars=sparsity))
            mask = self.fn_prune(sparsity=sparsity, param=param)
            return mask
        else:
            if verbose:
                print("{param_name:^30} | skipping".format(param_name=param_name))
            return None

    def prune(self, model, stage=0, update_masks=False, verbose=False):
        """
        prune models
        :param model: torch.nn.Module
        :param stage: int, the pruning stage, default=0
        :param update_masks: bool, whether update masks
        :param verbose: bool, whether to print the pruning details
        :return:
            void
        """
        update_masks = True if update_masks or len(self.masks) == 0 else False
        if verbose:
            print("=" * 89)
            print("Pruning Models")
            if len(self.masks) == 0:
                print("Initializing Masks")
            elif update_masks:
                print("Updating Masks")
            print("=" * 89)
            print("{name:^30} | stage | sparsity".format(name='param_name'))
        for param_name, param in model.named_parameters():
            if 'AuxLogits' not in param_name:
                # deal with googlenet
                if param.dim() > 1:
                    if update_masks:
                        mask = self.prune_param(param=param.data, param_name=param_name,
                                                stage=stage, verbose=verbose)
                        if mask is not None:
                            self.masks[param_name] = mask
                    else:
                        if param_name in self.masks:
                            mask = self.masks[param_name]
                            param.data.masked_fill_(mask, 0)
        if verbose:
            print("=" * 89)
