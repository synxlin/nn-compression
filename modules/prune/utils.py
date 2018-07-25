def get_sparsity(param):
    """

    :param param:
    :return:
    """
    mask = param.eq(0)
    return float(mask.sum()) / mask.numel()
