from .detectron_defaults import _C
from .config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.BACKBONE = CN()

# The backbone conv body to use
_C.BACKBONE.CONV_BODY = 'resnet'

# The eps of batch_norm layer
_C.BACKBONE.BN_EPS = 1e-5

# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
_C.BACKBONE.RESNET = CN()

# The number of layers in each block
# (2, 2, 2, 2) for resnet18 with basicblock
# (3, 4, 6, 3) for resnet34 with basicblock
# (3, 4, 6, 3) for resnet50
# (3, 4, 23, 3) for resnet101
# (3, 8, 36, 3) for resnet152
_C.BACKBONE.RESNET.LAYERS = (3, 4, 6, 3)

# Network stem width
_C.BACKBONE.RESNET.STEM_WIDTH = 64

# Network initial width
_C.BACKBONE.RESNET.WIDTH = 64

# Use bottleneck block, False for basicblock
_C.BACKBONE.RESNET.BOTTLENECK = True

# Use a aligned module in each block
_C.BACKBONE.RESNET.USE_ALIGN = False

# Use weight standardization
_C.BACKBONE.RESNET.USE_WS = False

# Place the stride 2 conv on the 3x3 filter.
# True for resnet-b
_C.BACKBONE.RESNET.STRIDE_3X3 = False

# Use a three (3 * 3) kernels head; False for (7 * 7) kernels head.
# True for resnet-c
_C.BACKBONE.RESNET.USE_3x3x3HEAD = False

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
# True for resnet-d
_C.BACKBONE.RESNET.AVG_DOWN = False

# Use SplAtConv2d in bottleneck.
# 2 for resnest
_C.BACKBONE.RESNET.RADIX = 1

# Type of 3x3 convolution layer in each block
# E.g., 'Conv2d', 'Conv2dWS', 'DeformConv', 'MDeformConv', ...
_C.BACKBONE.RESNET.STAGE_WITH_CONV = ('Conv2d', 'Conv2d', 'Conv2d', 'Conv2d')

# # Type of normalization
# E.g., 'Affine', 'BN', 'GN', 'MixBN', 'MixGN', ...
_C.BACKBONE.RESNET.NORM = 'BN'

# Type of context module in each block
# E.g., 'SE', 'GCB', ...
_C.BACKBONE.RESNET.STAGE_WITH_CTX = ('', '', '', '')

# Apply dilation in stage "c5"
_C.BACKBONE.RESNET.C5_DILATION = 1

# Freeze model weights before and including which block.
# Choices: [0, 2, 3, 4, 5]. O means not fixed. First conv and bn are defaults to
# be fixed.
_C.BACKBONE.RESNET.FREEZE_AT = 2
