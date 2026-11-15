# Adapted from YijieYang23/SkeleT-GCN, fb2b2b23, Apache-2.0.
# See NOTICE.md for attribution and modifications.
import torch.nn as nn

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
