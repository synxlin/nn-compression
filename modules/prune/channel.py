import torch


def find_channel_for_pruning(conv_feature, fn_next_layers, sparsity, method='greedy'):
