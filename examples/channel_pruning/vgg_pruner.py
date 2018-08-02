import re
import torch
from torchvision.models import VGG

from modules.prune import prune_channel


class VGGPruner(object):

    def __init__(self, rule):
        """
        Channel Pruner for VGG
        :param rule: str, path to the rule file, each line formats 'module_name sparsity'
                     list of tuple, [(module_name(str), sparsity(float))]
        """
        if isinstance(rule, str):
            content = map(lambda x: x.split(), open(rule).readlines())
            content = filter(lambda x: len(x) == 2, content)
            rule = list(map(lambda x: (x[0], float(x[1])), content))
        assert isinstance(rule, list) or isinstance(rule, tuple)

        self.rule = rule

    def get_param_sparsity(self, module_name):
        """
        get sparsity based on the name of module
        :param module_name: str, name of the module to prune
        :return:
            float, sparsity
        """
        rule_id = -1
        for idx, x in enumerate(self.rule):
            m = re.match(x[0], module_name)
            if m is not None and len(module_name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            sparsity = self.rule[rule_id][1]
            return sparsity
        else:
            return 1.0

    @staticmethod
    def get_prune_inputs(model):
        """
        get input args for prune() method of VGGPruner Class
        :param model: torch.nn.Module, model to prune
        :return:
            list of tuple, [(module_name, module, next_module, fn_input_feature, fn_next_input_feature), ...]
        """
        assert isinstance(model, VGG)
        features = model.features
        if isinstance(features, torch.nn.DataParallel):
            features = features.module
        classifier = model.classifier

        module_name_dict = dict()
        for n, m in model.named_modules():
            module_name_dict[m] = n

        conv_indices = []
        conv_modules = []
        conv_names = []
        for i, m in enumerate(features):
            if isinstance(m, torch.nn.modules.conv._ConvNd):
                conv_indices.append(i)
                conv_modules.append(m)
                conv_names.append(module_name_dict[m])

        fc_indices = []
        fc_modules = []
        fc_names = []
        for i, m in enumerate(classifier):
            if isinstance(m, torch.nn.Linear):
                fc_indices.append(i)
                fc_modules.append(m)
                fc_names.append(module_name_dict[m])

        def get_fn_conv_input_feature(idx):
            def fn(x):
                for seq_i in range(conv_indices[idx]):
                    x = features[seq_i](x)
                return x
            return fn

        def get_fn_next_input_feature(idx, module_indices, module_seq):
            def fn(x):
                for seq_i in range(module_indices[idx]+1, module_indices[idx+1]):
                    x = module_seq[seq_i](x)
                return x
            return fn

        prune_modules = []
        prune_module_names = []
        prune_module_fn = []
        prune_module_fn_next = []

        for i in range(len(conv_indices) - 1):
            prune_modules.append(conv_modules[i])
            prune_module_names.append(conv_names[i])
            prune_module_fn.append(get_fn_conv_input_feature(i))
            prune_module_fn_next.append(get_fn_next_input_feature(i, conv_indices, features))

        prune_modules.append(conv_modules[-1])
        prune_module_names.append(conv_names[-1])
        prune_module_fn.append(get_fn_conv_input_feature(-1))

        def fn_next_input_feature(x):
            for seq_i in range(conv_indices[-1]+1, len(features)):
                x = features[seq_i](x)
            x = x.view(x.size(0), -1)
            return x
        prune_module_fn_next.append(fn_next_input_feature)

        def get_fn_fc_input_feature(idx):
            def fn(x):
                x = features(x)
                x = x.view(x.size(0), -1)
                for seq_i in range(fc_indices[idx]):
                    x = classifier[seq_i](x)
                return x
            return fn

        for i in range(len(fc_indices) - 1):
            prune_modules.append(fc_modules[i])
            prune_module_names.append(fc_names[i])
            prune_module_fn.append(get_fn_fc_input_feature(i))
            prune_module_fn_next.append(get_fn_next_input_feature(i, fc_indices, classifier))

        prune_modules.append(fc_modules[-1])

        prune_inputs = []
        for i in range(len(prune_module_names)):
            prune_inputs.append((prune_module_names[i], prune_modules[i], prune_modules[i+1],
                                 prune_module_fn[i], prune_module_fn_next[i]))

        return prune_inputs

    def prune_module(self, module_name, module, next_module, fn_input_feature, fn_next_input_feature,
                     input, method='greedy', cpu=True, verbose=False):
        """

        :param module_name: str, the name of the module to prune
        :param module: torch.nn.Module, usually _ConvNd or Linear
        :param next_module: torch.nn.Module, the next _ConvNd or Linear module to "module"
        :param fn_input_feature: function, calculate input feature of "module" from the image
        :param fn_next_input_feature: function, calculate input feature of "next_module"
                                                from the output feature of "module"
        :param input: torch.tensor, input image of VGG, (batch_size, 3, 224, 224)
        :param method: str
                        'greedy': select one contributed to the smallest next feature after another
                        'lasso': select pruned channels by lasso regression
                        'random': randomly select
        :param cpu: bool, whether done in cpu for larger reconstruction batch size
        :return:
            void
        """
        sparsity = self.get_param_sparsity(module_name=module_name)
        if verbose:
            print("=" * 89)
            print("{param_name:^30} : {spars:.3f}".format(param_name=module_name, spars=sparsity))
        input_feature = fn_input_feature(input)
        prune_channel(sparsity=sparsity, module=module, next_module=next_module,
                      fn_next_input_feature=fn_next_input_feature,
                      input_feature=input_feature, method=method, cpu=cpu)
