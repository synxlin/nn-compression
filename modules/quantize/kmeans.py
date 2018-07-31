import numpy as np
from sklearn.cluster import KMeans
import torch


def quantize_k_means(param, k=16, codebook=None, guess='k-means++',
                     update_labels=False, re_quantize=False, **unused):
    """
    quantize using k-means clustering
    :param param:
    :param codebook: sklearn.cluster.KMeans, codebook of quantization, default=None
    :param k: int, the number of quantization level, default=16
    :param guess: str, initial quantization centroid generation method,
                       choose from 'linear', 'random', 'k-means++'
                  numpy.ndarray of shape (num_el, 1)
    :param update_labels: bool, whether to re-allocate the param elements to the latest centroids
    :param re_quantize: bool, whether to re-quantize the param
    :param unused: unused options
    :return:
        sklearn.cluster.KMeans, codebook of quantization
    """
    param_shape = param.size()
    num_el = param.numel()
    param_1d = param.view(num_el)

    if codebook is None or re_quantize:
        param_numpy = param_1d.view(num_el, 1).cpu().numpy()

        if guess == 'linear':
            guess = np.linspace(np.min(param_numpy), np.max(param_numpy), k)
            guess = guess.reshape(guess.size, 1)
        codebook = KMeans(n_clusters=k, init=guess, n_jobs=-1).fit(param_numpy)
        codebook.cluster_centers_ = torch.from_numpy(codebook.cluster_centers_).float()
        codebook.labels_ = torch.from_numpy(codebook.labels_).long()
        if param.is_cuda:
            codebook.cluster_centers_ = codebook.cluster_centers_.cuda(param.device)

    else:
        if update_labels:
            sorted_centers, indices = torch.sort(codebook.cluster_centers_, dim=0)
            boundaries = (sorted_centers[1:] + sorted_centers[:-1]) / 2
            sorted_labels = torch.ge(param_1d - boundaries, 0).long().sum(dim=0)
            codebook.labels_ = indices.index_select(0, sorted_labels).view(num_el)
        for i in range(k):
            codebook.cluster_centers_[i, 0] = param_1d[codebook.labels_ == i].mean()

    param_quantize = codebook.cluster_centers_[codebook.labels_].view(param_shape)
    if param.is_contiguous():
        param_quantize = param_quantize.contiguous()
    param.set_(param_quantize)

    return codebook


def quantize_k_means_fix_zeros(param, k=16, guess='k-means++', codebook=None,
                               update_labels=False, re_quantize=False, **unused):
    """
    quantize using k-means clustering while fixing the zeros
    :param param:
    :param codebook: sklearn.cluster.KMeans, codebook of quantization, default=None
    :param k: int, the number of quantization level, default=16
    :param guess: str, initial quantization centroid generation method,
                       choose from 'linear', 'random', 'k-means++'
    :param update_labels: bool, whether to re-allocate the param elements to the latest centroids
    :param re_quantize: bool, whether to re-quantize the param
    :param unused: unused options
    :return:
        sklearn.cluster.KMeans, codebook of quantization
    """
    param_shape = param.size()
    num_el = param.numel()
    param_1d = param.view(num_el)
    if codebook is not None:
        param_1d[codebook.labels_ == 0] = 0

    if codebook is None or re_quantize:
        param_numpy = param_1d.cpu().numpy()
        param_nz = param_numpy[param_numpy != 0]
        param_nz = param_nz.reshape(param_nz.size, 1)

        if guess == 'linear':
            guess = np.linspace(np.min(param_nz), np.max(param_nz), k - 1)  # one less cluster due to zero-fixed
            guess = guess.reshape(guess.size, 1)
        codebook = KMeans(n_clusters=k-1, init=guess, n_jobs=-1).fit(param_nz)  # one less cluster due to zero-fixed
        centers = codebook.cluster_centers_
        centers = np.append(0.0, centers)  # append zero as centroid[0]
        codebook.cluster_centers_ = centers.reshape(centers.size, 1)
        codebook.labels_ = codebook.predict(param_numpy.reshape(num_el, 1))
        codebook.cluster_centers_ = torch.from_numpy(codebook.cluster_centers_).float()
        codebook.labels_ = torch.from_numpy(codebook.labels_).long()
        if param.is_cuda:
            codebook.cluster_centers_ = codebook.cluster_centers_.cuda(param.device)

        # nonzero_indices = param_1d.nonzero()
        # nonzero_indices = nonzero_indices.view(nonzero_indices.numel())
        # codebook.cluster_centers_ = torch.from_numpy(codebook.cluster_centers_).float().cuda()
        # labels_ = torch.from_numpy(codebook.labels_).long().cuda().add_(1)
        # codebook.labels_ = labels_.new(num_el).zero_().index_copy_(0, nonzero_indices, labels_)

    else:
        if update_labels:
            sorted_centers, indices = torch.sort(codebook.cluster_centers_, dim=0)
            boundaries = (sorted_centers[1:] + sorted_centers[:-1]) / 2
            sorted_labels = torch.ge(param_1d - boundaries, 0).long().sum(dim=0)
            codebook.labels_ = indices.index_select(0, sorted_labels).view(num_el)
        for i in range(1, k):
            # not from (0, k), because we fix the zero centroid
            codebook.cluster_centers_[i, 0] = param_1d[codebook.labels_ == i].mean()

    param_quantize = codebook.cluster_centers_[codebook.labels_].view(param_shape)
    if not param.is_contiguous():
        param_quantize = param_quantize.contiguous()
    param.set_(param_quantize)

    return codebook
