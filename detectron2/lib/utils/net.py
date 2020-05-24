import itertools
import numpy as np

import torch

import detectron2.lib.ops as ops


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def convert_conv2convsamepadding_model(module, process_group=None, channel_last=False):
    mod = module
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        if isinstance(module.bias, torch.Tensor):
            bias = True
        else:
            bias = False
        mod = ops.Conv2dSamePadding(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                    module.dilation, module.groups, bias=bias)
        mod.weight.data = module.weight.data.clone().detach()
        if bias:
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_conv2convsamepadding_model(child, process_group=process_group,
                                                                channel_last=channel_last))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod


def convert_conv2convws_model(module, process_group=None, channel_last=False):
    mod = module
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        if isinstance(module.bias, torch.Tensor):
            bias = True
        else:
            bias = False
        mod = ops.Conv2dWS(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding,
                           module.dilation, module.groups, bias=bias)
        mod.weight.data = module.weight.data.clone().detach()
        if bias:
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_conv2convws_model(child, process_group=process_group, channel_last=channel_last))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod


def convert_bn2affine_model(module, process_group=None, channel_last=False, merge=True):
    """
    This function is learned from the NVIDIA/apex.
    It can be seen here:
    https://github.com/NVIDIA/apex/blob/master/apex/parallel/sync_batchnorm.py

    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `ops.AffineChannel2d`.
    """
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not isinstance(module, ops.MixtureBatchNorm2d):
        # print(module.weight.cpu().detach().numpy().shape)
        mod = ops.AffineChannel2d(module.num_features)
        mod.weight.data = module.weight.data.clone().detach()
        mod.bias.data = module.bias.data.clone().detach()
        freeze_params(mod)  # freeze affine params
        if merge:
            gamma = module.weight.data.clone().detach().numpy()
            beta = module.bias.data.clone().detach().numpy()
            mu = module.running_mean.data.clone().detach().numpy()
            var = module.running_var.data.clone().detach().numpy()
            eps = module.eps

            new_gamma = gamma / (np.power(var + eps, 0.5))  # new bn.weight
            new_beta = beta - gamma * mu / (np.power(var + eps, 0.5))  # new bn.bias

            mod.weight.data = torch.from_numpy(new_gamma)
            mod.bias.data = torch.from_numpy(new_beta)
    for name, child in module.named_children():
        mod.add_module(name, convert_bn2affine_model(child, process_group=process_group, channel_last=channel_last,
                                                     merge=merge))
    del module
    return mod


def convert_bn2syncbn_model(module, process_group=None):
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not isinstance(module, ops.MixtureBatchNorm2d):
        # frozen backbone bn layers which do not require grad
        for p in module.parameters():
            if not p.requires_grad:
                return convert_bn2frozenbn_model(mod)
        mod = ops.NaiveSyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
            # keep reuqires_grad unchanged
            mod.weight.requires_grad = module.weight.requires_grad
            mod.bias.requires_grad = module.bias.requires_grad
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        mod.add_module(name, convert_bn2syncbn_model(child, process_group=process_group))
    del module
    return mod


def convert_bn2frozenbn_model(module):
    bn_module = (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)
    mod = module
    if isinstance(module, bn_module):
        mod = ops.FrozenBatchNorm2d(module.num_features)
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
        mod.running_mean.data = module.running_mean.data
        mod.running_var.data = module.running_var.data
        mod.eps = module.eps
        freeze_params(mod)
    else:
        for name, child in module.named_children():
            new_child = convert_bn2frozenbn_model(child)
            if new_child is not child:
                mod.add_module(name, new_child)
    del module
    return mod


@torch.no_grad()
def update_bn_stats(model, data_loader, device, num_iters=200):
    assert model.training
    bn_layers = get_bn_modules(model)

    if len(bn_layers) == 0:
        return

    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    for ind, (images, targets, _) in enumerate(itertools.islice(data_loader, num_iters)):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        model(images, targets)

        for i, bn in enumerate(bn_layers):
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
            running_var[i] += (bn.running_var - running_var[i]) / (ind + 1)
    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind)
    )

    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]


def get_bn_modules(model):
    types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    bn_layers = [
        m
        for m in model.modules()
        if m.training and isinstance(m, types)
    ]
    return bn_layers


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


def mismatch_params_filter(s):
    l = []
    for i in s:
        if i.split('.')[-1] in ['num_batches_tracked', 'running_mean', 'running_var']:
            continue
        else:
            l.append(i)
    return l


def reduce_tensor(tensor, world_size=1):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt
