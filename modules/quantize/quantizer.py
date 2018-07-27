import re
from sklearn.cluster import KMeans
import torch

from .linear import quantize_linear, quantize_linear_fix_zeros
from .kmeans import quantize_k_means, quantize_k_means_fix_zeros


# TODO: fixed point arg
def quantize_vanilla(fix_zeros=True, method='k-means', guess=None, **options):
    """

    :param fix_zeros:
    :param method:
    :param guess:
    :param options:
    :return:
    """
    assert method in ['linear', 'k-means']
    # check guess options
    if method == 'linear':
        guess = 'linear'
    elif 'random' in guess:
        guess = 'random'
    elif 'uniform' in guess:
        guess = 'uniform'
    elif isinstance(guess, str):
        guess = 'k-means++'
    else:
        assert torch.is_tensor(guess)
        guess = guess.view(guess.numel(), 1).cpu().numpy()
    if method == 'linear':
        if fix_zeros:
            return quantize_linear_fix_zeros(**options)
        else:
            return quantize_linear(**options)
    elif fix_zeros:
        return quantize_k_means_fix_zeros(guess=guess, **options)
    else:
        return quantize_k_means(guess=guess, **options)


# TODO: fixed point arg
class VanillaQuantizer(object):
    """

    """
    def __init__(self, rule, fix_zeros=True):
        """

        :param rule:
        :param fix_zeros:
        """
        if isinstance(rule, str):
            content = map(lambda x: x.split(), open(rule).readlines())
            content = filter(lambda x: len(x) == 3, content)
            rule = list(map(lambda x: (x[0], int(x[1]), x[2]), content))
        assert isinstance(rule, list)
        self.rule = rule

        self.codebooks = dict()
        self.fix_zeros = fix_zeros
        self.quantize = quantize_vanilla

        print("Initializing Vanilla Quantizer\nRules:\n{}".format(self.rule))

    def load_state_dict(self, state_dict):
        """

        :param state_dict:
        :return:
        """
        self.rule = state_dict['rule']
        self.fix_zeros = state_dict['fix_zeros']
        self.codebooks = dict()
        for name, codebook in state_dict['codebooks'].items():
            if codebook['method'] == 'k-means':
                self.codebooks[name] = KMeans().set_params(**codebook['params'])
                self.codebooks[name].cluster_centers_ = codebook['centers']
                self.codebooks[name].labels_ = codebook['labels']
            else:
                self.codebooks[name] = codebook
        return self

    def state_dict(self):
        """

        :return:
        """
        state_dict = dict()
        state_dict['rule'] = self.rule
        state_dict['fix_zeros'] = self.fix_zeros
        codebooks = dict()
        for name, codebook in self.codebooks.items():
            if isinstance(codebook, KMeans):
                codebooks[name] = {
                    'params': codebook.get_params(),
                    'centers': codebook.cluster_centers_,
                    'labels': codebook.labels_,
                }
            else:
                codebooks[name] = codebook
        state_dict['codebooks'] = codebooks
        return state_dict

    def quantize_param(self, param, param_name, **quantize_options):
        """

        :param param:
        :param param_name:
        :param quantize_options:
        :return:
        """
        rule_id = -1
        for idx, x in enumerate(self.rule):
            m = re.match(x[0], param_name)
            if m is not None and len(param_name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            k = self.rule[rule_id][1]
            guess = self.rule[rule_id][2]
            codebook = self.codebooks.get(param_name)
            if codebook is None:
                print('{}:\t\tquantize level: {}'.format(param_name, k))
            codebook = self.quantize(fix_zeros=self.fix_zeros, guess=guess,
                                     param=param, codebook=codebook, k=k,
                                     **quantize_options)
            return codebook
        else:
            print('{}:\t\tskipping'.format(param_name))
            return None

    def quantize_model(self, model, update_centers=False, update_labels=False, re_quantize=False):
        """

        :param model:
        :param update_centers:
        :param update_labels:
        :param re_quantize:
        :return:
        """
        for param_name, param in model.named_parameters():
            if param.dim() > 1:
                codebook = self.quantize_param(param.data, param_name, update_centers=update_centers,
                                               update_labels=update_labels, re_quantize=re_quantize)
                if codebook is not None:
                    self.codebooks[param_name] = codebook
