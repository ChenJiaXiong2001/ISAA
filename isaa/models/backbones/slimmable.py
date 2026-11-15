"""Weight-sharing execution of the existing full GCNs at stage-wise widths.

CTR outputs retain prefixes of *each* temporal branch, rather than dropping
the pooling/pointwise branches by slicing the concatenated tensor's prefix.
Only the selected convolution kernels and relation channels are executed.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import prepare_masked_input, downsample_mask

STAGE_LAYERS = (4, 3, 3)
STAGE_OF_LAYER = (0, 0, 0, 0, 1, 1, 1, 2, 2, 2)


def indices(width, maximum, device, branches=1):
    if width == maximum:
        return torch.arange(maximum, device=device)
    return torch.cat([torch.arange(width // branches, device=device)
                      + branch * (maximum // branches) for branch in range(branches)])


def conv(module, x, outputs, inputs):
    weight = module.weight.index_select(0, outputs).index_select(1, inputs)
    bias = None if module.bias is None else module.bias.index_select(0, outputs)
    return F.conv2d(x, weight, bias, module.stride, module.padding, module.dilation, module.groups)


def batch_norm(module, x, ids, update):
    mean = module.running_mean.index_select(0, ids)
    variance = module.running_var.index_select(0, ids)
    training = bool(update and module.training)
    momentum = module.momentum
    if training:
        module.num_batches_tracked.add_(1)
        if momentum is None:
            momentum = 1.0 / float(module.num_batches_tracked)
    y = F.batch_norm(x, mean, variance, module.weight.index_select(0, ids),
                     module.bias.index_select(0, ids), training,
                     0.0 if momentum is None else momentum, module.eps)
    if training:
        with torch.no_grad():
            module.running_mean.index_copy_(0, ids, mean)
            module.running_var.index_copy_(0, ids, variance)
    return y


def temporal(module, x, inputs, outputs, update):
    return batch_norm(module.bn, conv(module.conv, x, outputs, inputs), outputs, update)


def ctr_block(block, x, inputs, outputs, update):
    gcn = block.gcn1
    a = gcn.PA if gcn.adaptive else gcn.A.to(x)
    value = None
    for partition, module in enumerate(gcn.convs):
        relation_width = 8 if module.conv1.in_channels in (3, 6, 9) else len(inputs) // 8
        relations = torch.arange(relation_width, device=x.device)
        q = conv(module.conv1, x, relations, inputs).mean(-2)
        k = conv(module.conv2, x, relations, inputs).mean(-2)
        feature = conv(module.conv3, x, outputs, inputs)
        adjacency = conv(module.conv4, torch.tanh(q.unsqueeze(-1) - k.unsqueeze(-2)),
                         outputs, relations) * gcn.alpha + a[partition][None, None]
        branch = torch.einsum("bcuv,bctv->bctu", adjacency, feature)
        value = branch if value is None else value + branch
    if isinstance(gcn.down, nn.Sequential):
        residual = batch_norm(gcn.down[1], conv(gcn.down[0], x, outputs, inputs), outputs, update)
    else:
        # Within one stage, input/output selections are identical, as in the full GCN.
        residual = gcn.down(x)
    spatial = F.relu(batch_norm(gcn.bn, value, outputs, update) + residual)
    tcn = block.tcn1
    selected = torch.arange(len(outputs) // tcn.num_branches, device=x.device)
    branches = []
    for branch in tcn.branches:
        y = conv(branch[0], spatial, selected, outputs)
        y = batch_norm(branch[1], y, selected, update)
        if len(branch) > 2:
            y = F.relu(y)
            if hasattr(branch[3], "conv"):
                y = temporal(branch[3], y, selected, selected, update)
            else:
                y = branch[3](y)
                y = batch_norm(branch[4], y, selected, update)
        branches.append(y)
    if hasattr(block.residual, "conv"):
        residual = temporal(block.residual, x, inputs, outputs, update)
    else:
        residual = block.residual(x)
    return F.relu(torch.cat(branches, 1) + residual)


def st_block(block, x, a, inputs, outputs, update):
    maximum = block.tcn[0].num_features
    graph_outputs = torch.cat([outputs + partition * maximum for partition in range(3)])
    y = conv(block.gcn.conv, x, graph_outputs, inputs)
    b, _, t, v = y.shape
    y = torch.einsum("nkctv,kvw->nctw", y.reshape(b, 3, len(outputs), t, v), a).contiguous()
    y = F.relu(batch_norm(block.tcn[0], y, outputs, update))
    y = conv(block.tcn[2], y, outputs, outputs)
    y = batch_norm(block.tcn[3], y, outputs, update)
    y = F.dropout(y, p=block.tcn[4].p, training=block.training)
    if isinstance(block.residual, nn.Sequential):
        residual = batch_norm(block.residual[1], conv(block.residual[0], x, outputs, inputs), outputs, update)
    else:
        residual = block.residual(x)
    return F.relu(y + residual)


def forward_slim(encoder, x, mask, widths, *, kind, gates=None, update_bn=False,
                 block_executor=None):
    x, mask = prepare_masked_input(x, mask, encoder.in_channels, encoder.num_point)
    b, c, t, v = x.shape
    inputs = torch.arange(c, device=x.device)
    x = x.permute(0, 3, 1, 2).reshape(b, v * c, t)
    x = batch_norm(encoder.data_bn, x, torch.arange(v * c, device=x.device), update_bn)
    x = x.reshape(b, v, c, t).permute(0, 2, 3, 1).contiguous().masked_fill(~mask, 0)
    layers = encoder.blocks if kind == "ctr" else encoder.st_gcn_networks
    for layer, (block, stride, maximum) in enumerate(zip(layers, encoder.strides, encoder.channels)):
        stage = STAGE_OF_LAYER[layer]
        outputs = indices(widths[stage], maximum, x.device, 4 if kind == "ctr" else 1)
        if block_executor is not None:
            adjacency = None if kind == "ctr" else encoder.A.to(x) * encoder.edge_importance[layer]
            x = block_executor(block, x, inputs, outputs, adjacency, layer)
        elif kind == "ctr":
            x = ctr_block(block, x, inputs, outputs, update_bn)
        else:
            x = st_block(block, x, encoder.A.to(x) * encoder.edge_importance[layer], inputs, outputs, update_bn)
        mask = downsample_mask(mask, stride)
        x = x.masked_fill(~mask, 0)
        if gates is not None:
            x = x * gates[stage].index_select(1, outputs)[:, :, None, None]
        inputs = outputs
    # Preserve the existing fusion/projection contract; GCN kernels above are narrow.
    padded = x.new_zeros((b, encoder.channels[-1], x.shape[2], v))
    return padded.index_copy(1, inputs, x), mask
