from .vanilla import *


class Pruner(object):
    def __init__(self, rule, method='vanilla', granularity='element'):
        assert method in ['vanilla', 'channel']

        if isinstance(rule, str):
            content = map(lambda x: x.split(), open(rule).readlines())
            content = filter(lambda x: len(x) == 2, content)
            rule = list(map(lambda x: (x[0], list(map(float, x[1].split(',')))), content))
        assert isinstance(rule, list)

        self.method = method
        if self.method == 'vanilla':
            if granularity == 'element':
                self.prune = prune_vanilla_elementwise
            elif granularity == 'kernel':
                self.prune = prune_vanilla_kernelwise
            elif granularity == 'filter':
                self.prune = prune_vanilla_filterwise
            else:
                raise NotImplementedError
            self.granularity = granularity
        else:
            self.granularity = 'channel'



