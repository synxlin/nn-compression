import torch


def replicate(network, keep_param=False):
    assert isinstance(network, torch.nn.Module)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    param_copies = [torch.nn.Parameter(param.detach().clone()) for param in params]

    buffers = list(network._all_buffers())
    buffer_indices = {buf: idx for idx, buf in enumerate(buffers)}
    buffer_copies = [buffer.clone() for buffer in buffers]

    modules = list(network.modules())
    module_indices = {}
    module_copies = []

    for i, module in enumerate(modules):
        module_indices[module] = i
        replica = module.__new__(type(module))
        replica.__dict__ = module.__dict__.copy()
        replica._parameters = replica._parameters.copy()
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        module_copies.append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                replica = module_copies[i]
                replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                replica = module_copies[i]
                replica._modules[key] = module_copies[module_idx]
        for key, param in module._parameters.items():
            if param is None:
                replica = module_copies[i]
                replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                replica = module_copies[i]
                replica._parameters[key] = param_copies[param_idx]
        for key, buf in module._buffers.items():
            if buf is None:
                replica = module_copies[i]
                replica._buffers[key] = None
            else:
                buffer_idx = buffer_indices[buf]
                replica = module_copies[i]
                replica._buffers[key] = buffer_copies[buffer_idx]

    return module_copies[0]
