import torch
import math
import numpy as np
from sklearn.cluster import KMeans

magic_percentile = 0.001


def quantize_linear(param, k=16, **unused):
    num_el = param.numel()
    kth = int(math.ceil(num_el * magic_percentile))
    param_flatten = param.view(num_el)
    param_min, _ = torch.topk(param_flatten, kth, dim=0, largest=False, sorted=False)
    param_min = param_min.max()
    param_max, _ = torch.topk(param_flatten, kth, dim=0, largest=True, sorted=False)
    param_max = param_max.min()
    step = (param_max - param_min) / (k - 1)
    param.clamp_(param_min, param_max).sub_(param_min).div_(step).round_().mul_(step).add_(param_min)
    codebook = torch.tensor(list(set(param_flatten.cpu().tolist())))
    # codebook = torch.linspace(param_min, param_max, k)
    return codebook


def quantize_linear_fix_zeros(param, k=16, **unused):
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
    codebook = torch.tensor(list(set(param_flatten.cpu().tolist())))
    # codebook = torch.linspace(param_min, param_max, k - 1)
    return codebook


def quantize_k_means(param, codebook=None, k=16, guess=None, update_centers=False,
             update_labels=False, re_quantize=False):
    param_shape = param.size()
    num_el = param.numel()
    param_1d = param.view(num_el)

    if codebook is None or re_quantize:
        # if codebook is None:
        #     print("------ creating codebook ------")
        # else:
        #     print("------   re-quantizing   ------")
        param_numpy = param_1d.view(num_el, 1).cpu().numpy()

        if guess is 'uniform':
            guess = np.linspace(np.min(param_numpy), np.max(param_numpy), k)
            guess = guess.reshape(guess.size, 1)
        codebook = KMeans(n_clusters=k, init=guess, n_jobs=-1).fit(param_numpy)
        codebook.cluster_centers_ = torch.from_numpy(codebook.cluster_centers_).float()
        codebook.labels_ = torch.from_numpy(codebook.labels_).long()

    if update_centers or update_labels:
        if update_labels:
            cluster_centers = codebook.cluster_centers_
            if param.is_cuda:
                cluster_centers = cluster_centers.cuda(param.device)
            sorted_centers, indices = torch.sort(cluster_centers, dim=0)
            boundaries = (sorted_centers[1:] + sorted_centers[:-1]) / 2
            boundary_mask = torch.ge(param_1d - boundaries, 0).long()
            sorted_labels = boundary_mask.sum(dim=0)
            new_labels_ = indices.index_select(0, sorted_labels).view(num_el)
            codebook.labels_ = new_labels_.cpu()
        for i in range(k):
            codebook.cluster_centers_[i, 0] = param_1d[codebook.labels_ == i].mean()

    param_quantize = codebook.cluster_centers_[codebook.labels_].view(param_shape)
    if param.is_cuda:
        param_quantize = param_quantize.cuda(param.device)
    else:
        param_quantize = param_quantize.contiguous()
    param.set_(param_quantize)

    return codebook


def quantize_k_means_fix_zeros(param, codebook=None, k=16, guess=None, update_centers=False,
                               update_labels=False, re_quantize=False):
    param_shape = param.size()
    num_el = param.numel()
    param_1d = param.view(num_el)
    if codebook is not None:
        param_1d[codebook.labels_ == 0] = 0

    nonzero_indices = param_1d.nonzero()
    nonzero_indices = nonzero_indices.view(nonzero_indices.numel())

    if codebook is None or re_quantize:
        # if codebook is None:
        #     print("------ creating codebook ------")
        # else:
        #     print("------   re-quantizing   ------")
        param_numpy = param_1d.cpu().numpy()
        param_nz = param_numpy[param_numpy != 0]
        param_nz = param_nz.reshape(param_nz.size, 1)

        if guess is 'uniform':
            guess = np.linspace(np.min(param_nz), np.max(param_nz), k-1)  # one less cluster due to zero-fixed
            guess = guess.reshape(guess.size, 1)
        codebook = KMeans(n_clusters=k-1, init=guess, n_jobs=-1).fit(param_nz)  # one less cluster due to zero-fixed
        centers = codebook.cluster_centers_
        centers = np.append(0.0, centers) # append zero as centroid[0]
        codebook.cluster_centers_ = centers.reshape(centers.size, 1)
        codebook.cluster_centers_ = torch.from_numpy(codebook.cluster_centers_).float().cuda()
        labels_ = torch.from_numpy(codebook.labels_).long().cuda().add_(1)
        codebook.labels_ = labels_.new(numel).zero_().index_copy_(0, nonzero_indices, labels_)

    if update_centers or update_labels:
        if update_labels:
            sorted_centers, indices = torch.sort(codebook.cluster_centers_, dim=0)
            boundaries = (sorted_centers[1:] + sorted_centers[:-1]) / 2
            boundary_mask = torch.ge(param_1d - boundaries, 0)
            sorted_labels = boundary_mask.sum(dim=0).long()
            new_labels_ = indices.index_select(0, sorted_labels).view(numel)
            #print("#weights changed grouping =", torch.ne(codebook.labels_, new_labels_).sum())
            #print(" total #weights =", numel)
            codebook.labels_ = new_labels_
        for i in range(1, k): #not from (0, k), because we fix the zero centroid
            codebook.cluster_centers_[i, 0] = param_1d[codebook.labels_ == i].mean()

    param_quantize = codebook.cluster_centers_[codebook.labels_].view(param_shape).contiguous()
    param.set_(param_quantize)

    return codebook


def quantize_func(fix_zeros=True, guess=None, **options):
    # check guess options
    if 'linear' in guess:
        guess = 'linear'
    elif 'random' in guess:
        guess = 'random'
    elif 'uniform' in guess:
        guess = 'uniform'
    elif isinstance(guess, str):
        guess = 'k-means++'
    else:
        assert isinstance(guess, np.ndarray)
    if guess == 'linear':
        if fix_zeros:
            return quantize_linear_fix_zeros_in_place(**options)
        else:
            return quantize_linear_in_place(**options)
    elif fix_zeros:
        return quantize_fix_zeros_in_place(guess=guess, **options)
    else:
        return quantize_in_place(guess=guess, **options)