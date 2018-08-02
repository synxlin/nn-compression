import re
from sklearn.cluster import KMeans

from modules.quantize.fixed_point import quantize_fixed_point
from modules.quantize.linear import quantize_linear, quantize_linear_fix_zeros
from modules.quantize.kmeans import quantize_k_means, quantize_k_means_fix_zeros


def vanilla_quantize(method='k-means', fix_zeros=True, **options):
    """
    returns quantization function based on the options
    :param fix_zeros: bool, whether to fix zeros in the param
    :param method: str, quantization method, choose from 'linear', 'k-means'
    :param bit_length: int, bit length of fixed point param, default=8
    :param bit_length_integer: int, bit length of integer part
                                    of fixed point param, default=0
    :param k: int, the number of quantization level, default=16
    :param codebook: sklearn.cluster.KMeans, codebook of quantization, default=None
    :param guess: str, initial quantization centroid generation method,
                       choose from 'linear', 'random', 'k-means++'
                  numpy.ndarray of shape (num_el, 1)
    :param update_labels: bool, whether to re-allocate the param elements
                                to the latest centroids when using k-means
    :param re_quantize: bool, whether to re-quantize the param when using k-means
    :return:
        codebook
    """
    if method == 'k-means':
        if fix_zeros:
            return quantize_k_means_fix_zeros(**options)
        else:
            return quantize_k_means(**options)
    elif method == 'linear':
        if fix_zeros:
            return quantize_linear_fix_zeros(**options)
        else:
            return quantize_linear(**options)
    else:
        return quantize_fixed_point(**options)


class VanillaQuantizer(object):

    def __init__(self, rule=None, fix_zeros=True):
        """
        Quantizer class for quantization
        :param rule: str, path to the rule file, each line formats
                        'param_name method bit_length initial_guess_or_bit_length_of_integer'
                     list of tuple,
                        [(param_name(str), method(str), bit_length(int),
                          initial_guess(str)_or_bit_length_of_integer(int))]
        :param fix_zeros: whether to fix zeros when quantizing
        """
        if isinstance(rule, str):
            content = map(lambda x: x.split(), open(rule).readlines())
            content = filter(lambda x: len(x) == 4, content)
            rule = list(map(lambda x: (x[0], x[1], int(x[2]),
                                       int(x[3]) if x[1] == 'fixed_point' else x[3]),
                            content))
        assert isinstance(rule, list) or isinstance(rule, tuple) or rule is None
        self.rule = rule

        self.codebooks = dict()
        self.fix_zeros = fix_zeros
        self.fn_quantize = vanilla_quantize

        print("=" * 89)
        if self.rule is None:
            print("Initializing Quantizer WITHOUT rules")
        else:
            print("Initializing Quantizer with rules:")
            for r in self.rule:
                print(r)
        print("=" * 89)

    def load_state_dict(self, state_dict):
        """
        Recover Quantizer
        :param state_dict: dict, a dictionary containing a whole state of the Quantizer
        :return:
            Quantizer
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
        print("=" * 89)
        print("Customizing Quantizer with rules:")
        for r in self.rule:
            print(r)
        print("=" * 89)

    def state_dict(self):
        """
        Returns a dictionary containing a whole state of the Quantizer
        :return: dict, a dictionary containing a whole state of the Quantizer
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
                    'method': 'k-means'
                }
            else:
                codebooks[name] = codebook
        state_dict['codebooks'] = codebooks
        return state_dict

    def quantize_param(self, param, param_name, verbose=False, **quantize_options):
        """
        quantize param
        :param param: torch.(cuda.)tensor
        :param param_name: str, name of param
        :param update_labels: bool, whether to re-allocate the param elements
                                    to the latest centroids when using k-means
        :param re_quantize: bool, whether to re-quantize the param when using k-means
        :param verbose: bool, whether to print quantize details
        :return:
            dict, {'centers_': torch.tensor}, codebook of linear quantization
            sklearn.cluster.KMeans, codebook of k-means quantization
        """
        rule_id = -1
        for idx, r in enumerate(self.rule):
            m = re.match(r[0], param_name)
            if m is not None and len(param_name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            method = self.rule[rule_id][1]
            bit_length = self.rule[rule_id][2]
            k = 2 ** bit_length
            guess = self.rule[rule_id][3]
            bit_length_integer = guess
            if k <= 0:
                if verbose:
                    print("{param_name:^30} | skipping".format(param_name=param_name))
                return None
            codebook = self.codebooks.get(param_name)
            if verbose:
                if codebook is None:
                    print("{param_name:^30} | {bit_length:2d} bit | initializing".
                          format(param_name=param_name, bit_length=bit_length))
                elif method == 'k-means':
                    if quantize_options.get('re_quantize'):
                        print("{param_name:^30} | {bit_length:2d} bit | re-quantizing".
                              format(param_name=param_name, bit_length=bit_length))
                    elif quantize_options.get('update_labels'):
                        print("{param_name:^30} | {bit_length:2d} bit | updating labels and centroids".
                              format(param_name=param_name, bit_length=bit_length))
                    else:
                        print("{param_name:^30} | {bit_length:2d} bit | updating centroids only".
                              format(param_name=param_name, bit_length=bit_length))
                else:
                    print("{param_name:^30} | {bit_length:2d} bit | re-quantizing".
                          format(param_name=param_name, bit_length=bit_length))
            codebook = self.fn_quantize(method=method, fix_zeros=self.fix_zeros,
                                        param=param, bit_length=bit_length,
                                        bit_length_integer=bit_length_integer,
                                        k=k, guess=guess, codebook=codebook,
                                        **quantize_options)
            return codebook
        else:
            if verbose:
                print("{param_name:^30} | skipping".format(param_name=param_name))
            return None

    def quantize(self, model, update_labels=False, re_quantize=False, verbose=False):
        """
        quantize model
        :param model: torch.nn.module
        :param update_labels: bool, whether to re-allocate the param elements
                                    to the latest centroids when using k-means
        :param re_quantize: bool, whether to re-quantize the param when using k-means
        :param verbose: bool, whether to print quantize details
        :return:
            void
        """
        if verbose:
            print("=" * 89)
            print("Quantizing Model")
            print("=" * 89)
            print("{name:^30} | qz bit | state".format(name='param_name'))
        for param_name, param in model.named_parameters():
            if param.dim() > 1:
                codebook = self.quantize_param(param.data, param_name, verbose=verbose,
                                               update_labels=update_labels,
                                               re_quantize=re_quantize)
                if codebook is not None:
                    self.codebooks[param_name] = codebook
        if verbose:
            print("=" * 89)
