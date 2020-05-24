import math

import torch.nn as nn
import torch.nn.functional as F

import detectron2.lib.backbone.resnet as res
import detectron2.lib.ops as ops
from detectron2.lib.layers import make_norm, BasicBlock, Bottleneck, AlignedBottleneck
from detectron2.lib.utils.net import convert_conv2convws_model, freeze_params
from detectron2.lib.utils.poolers import Pooler
from .build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec


class ResNet(res.ResNet):
    def __init__(self, cfg, stride=32):
        """ Constructor
        """
        super(ResNet, self).__init__()
        if cfg.BACKBONE.RESNET.USE_ALIGN:
            block = AlignedBottleneck
        else:
            if cfg.BACKBONE.RESNET.BOTTLENECK:
                block = Bottleneck  # not use the original Bottleneck module
            else:
                block = BasicBlock
        stem_width = cfg.BACKBONE.RESNET.STEM_WIDTH
        layers = cfg.BACKBONE.RESNET.LAYERS[:int(math.log(stride, 2)) - 1]
        stage_with_conv = cfg.BACKBONE.RESNET.STAGE_WITH_CONV
        norm = cfg.BACKBONE.RESNET.NORM
        stage_with_ctx = cfg.BACKBONE.RESNET.STAGE_WITH_CTX
        c5_dilation = cfg.BACKBONE.RESNET.C5_DILATION
        self._out_features = cfg.MODEL.RESNETS.OUT_FEATURES

        self.expansion = block.expansion
        self.use_3x3x3stem = cfg.BACKBONE.RESNET.USE_3x3x3HEAD
        self.stride_3x3 = cfg.BACKBONE.RESNET.STRIDE_3X3
        self.avg_down = cfg.BACKBONE.RESNET.AVG_DOWN
        self.base_width = cfg.BACKBONE.RESNET.WIDTH
        self.radix = cfg.BACKBONE.RESNET.RADIX
        self.norm = norm
        self.layers = layers
        self.stride = stride
        self.freeze_at = cfg.BACKBONE.RESNET.FREEZE_AT

        self._out_feature_strides = {"stem": 4}
        self._out_feature_channels = {"stem": cfg.MODEL.RESNETS.STEM_OUT_CHANNELS}

        self.inplanes = stem_width
        if not self.use_3x3x3stem:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
            self.bn1 = make_norm(self.inplanes, norm=norm.replace('Mix', ''))
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes // 2, 3, 2, 1, bias=False)
            self.bn1 = make_norm(self.inplanes // 2, norm=norm.replace('Mix', ''))
            self.conv2 = nn.Conv2d(self.inplanes // 2, self.inplanes // 2, 3, 1, 1, bias=False)
            self.bn2 = make_norm(self.inplanes // 2, norm=norm.replace('Mix', ''))
            self.conv3 = nn.Conv2d(self.inplanes // 2, self.inplanes, 3, 1, 1, bias=False)
            self.bn3 = make_norm(self.inplanes, norm=norm.replace('Mix', ''))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1, conv=stage_with_conv[0], ctx=stage_with_ctx[0])
        self._out_feature_strides["res2"] = 4
        self._out_feature_channels["res2"] = 256
        self.layer2 = self._make_layer(block, 128, layers[1], 2, conv=stage_with_conv[1], ctx=stage_with_ctx[1])
        self._out_feature_strides["res3"] = 8
        self._out_feature_channels["res3"] = 512
        self.layer3 = self._make_layer(block, 256, layers[2], 2, conv=stage_with_conv[2], ctx=stage_with_ctx[2])
        self._out_feature_strides["res4"] = 16
        self._out_feature_channels["res4"] = 1024

        if len(layers) == 4:
            if c5_dilation != 1:
                c5_stride = 1
            else:
                c5_stride = 2
            self.layer4 = self._make_layer(block, 512, layers[3], c5_stride, dilation=c5_dilation,
                                           conv=stage_with_conv[3], ctx=stage_with_ctx[3])
            self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32. * c5_dilation]
            self._out_feature_strides["res5"] = 32
            self._out_feature_channels["res5"] = 2048
        else:
            del self.layer4
            self.spatial_scale = [1 / 4., 1 / 8., 1 / 16.]

        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        del self.avgpool
        del self.fc
        self._init_weights()
        self._init_modules()

    def _init_modules(self):
        assert self.freeze_at in [0, 2, 3, 4, 5]  # cfg.BACKBONE.RESNET.FREEZE_AT: 2
        assert self.freeze_at <= len(self.layers) + 1
        if self.freeze_at > 0:
            freeze_params(getattr(self, 'conv1'))
            freeze_params(getattr(self, 'bn1'))
            if self.use_3x3x3stem:
                freeze_params(getattr(self, 'conv2'))
                freeze_params(getattr(self, 'bn2'))
                freeze_params(getattr(self, 'conv3'))
                freeze_params(getattr(self, 'bn3'))
        for i in range(1, self.freeze_at):
            freeze_params(getattr(self, 'layer%d' % i))
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, ops.AffineChannel2d) else None)

    def train(self, mode=True):
        # Override train mode
        self.training = mode
        if self.freeze_at < 1:
            getattr(self, 'conv1').train(mode)
            getattr(self, 'bn1').train(mode)
            if self.use_3x3x3stem:
                getattr(self, 'conv2').train(mode)
                getattr(self, 'bn2').train(mode)
                getattr(self, 'conv3').train(mode)
                getattr(self, 'bn3').train(mode)
        for i in range(self.freeze_at, len(self.layers) + 1):
            if i == 0:
                continue
            getattr(self, 'layer%d' % i).train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def forward(self, x):
        outputs = {}
        if not self.use_3x3x3stem:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        outputs["res2"] = x2
        x3 = self.layer2(x2)
        outputs["res3"] = x3
        x4 = self.layer3(x3)
        outputs["res4"] = x4

        if len(self.layers) == 4:
            x5 = self.layer4(x4)
            outputs["res5"] = x5
            return outputs
            # return [x2, x3, x4, x5]
        else:
            return [x2, x3, x4]


class ResNet_C5_Head(res.ResNet):
    def __init__(self, cfg, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]

        if cfg.BACKBONE.RESNET.USE_ALIGN:
            block = res.AlignedBottleneck
        else:
            if cfg.BACKBONE.RESNET.BOTTLENECK:
                block = res.Bottleneck  # not use the original Bottleneck module
            else:
                block = res.BasicBlock
        layers = cfg.BACKBONE.RESNET.LAYERS
        stage_with_conv = cfg.BACKBONE.RESNET.STAGE_WITH_CONV
        norm = cfg.BACKBONE.RESNET.NORM
        stage_with_ctx = cfg.BACKBONE.RESNET.STAGE_WITH_CTX
        c5_dilation = cfg.BACKBONE.RESNET.C5_DILATION

        self.expansion = block.expansion
        self.stride_3x3 = cfg.BACKBONE.RESNET.STRIDE_3X3
        self.avg_down = cfg.BACKBONE.RESNET.AVG_DOWN
        self.base_width = cfg.BACKBONE.RESNET.WIDTH
        self.radix = cfg.BACKBONE.RESNET.RADIX
        self.norm = norm

        method = cfg.FAST_RCNN.ROI_XFORM_METHOD
        resolution = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        self.inplanes = self.dim_in
        c5_stride = min(resolution) // 7
        self.layer4 = self._make_layer(block, 512, layers[3], c5_stride, dilation=c5_dilation,
                                       conv=stage_with_conv[3], ctx=stage_with_ctx[3])
        self.dim_out = self.stage_out_dim[-1]

        del self.conv1
        del self.bn1
        del self.relu
        del self.maxpool
        del self.layer1
        del self.layer2
        del self.layer3
        del self.avgpool
        del self.fc
        self._init_weights()

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.layer4(x)

        return x


class ResNet_2mlp_Head(res.ResNet):
    def __init__(self, cfg, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]

        if cfg.BACKBONE.RESNET.USE_ALIGN:
            block = res.AlignedBottleneck
        else:
            if cfg.BACKBONE.RESNET.BOTTLENECK:
                block = res.Bottleneck  # not use the original Bottleneck module
            else:
                block = res.BasicBlock
        layers = cfg.BACKBONE.RESNET.LAYERS
        stage_with_conv = cfg.BACKBONE.RESNET.STAGE_WITH_CONV
        norm = cfg.BACKBONE.RESNET.NORM
        stage_with_ctx = cfg.BACKBONE.RESNET.STAGE_WITH_CTX
        c5_dilation = cfg.BACKBONE.RESNET.C5_DILATION

        self.expansion = block.expansion
        self.stride_3x3 = cfg.BACKBONE.RESNET.STRIDE_3X3
        self.avg_down = cfg.BACKBONE.RESNET.AVG_DOWN
        self.base_width = cfg.BACKBONE.RESNET.WIDTH
        self.radix = cfg.BACKBONE.RESNET.RADIX
        self.norm = norm

        self.inplanes = self.dim_in
        c5_stride = 2 if c5_dilation == 1 else 1
        self.layer4 = self._make_layer(block, 512, layers[3], c5_stride, dilation=c5_dilation,
                                       conv=stage_with_conv[3], ctx=stage_with_ctx[3])
        self.conv_new = nn.Sequential(
            nn.Conv2d(512 * self.expansion, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.dim_in = 256
        method = cfg.FAST_RCNN.ROI_XFORM_METHOD
        resolution = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        input_size = self.dim_in * resolution[0] * resolution[1]
        mlp_dim = cfg.FAST_RCNN.MLP_HEAD.MLP_DIM
        self.fc1 = nn.Linear(input_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.dim_out = mlp_dim

        del self.conv1
        del self.bn1
        del self.relu
        del self.maxpool
        del self.layer1
        del self.layer2
        del self.layer3
        del self.avgpool
        del self.fc
        self._init_weights()

    def forward(self, x, proposals):
        x = self.layer4(x[0])
        x = self.conv_new(x)

        x = self.pooler([x], proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    model = ResNet(cfg)
    if cfg.BACKBONE.RESNET.USE_WS:
        model = convert_conv2convws_model(model)
    return model
    # need registration of new blocks/stems?
    # norm = cfg.MODEL.RESNETS.NORM
    # stem = BasicStem(
    #     in_channels=input_shape.channels,
    #     out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
    #     norm=norm,
    # )
    #
    # # fmt: off
    # freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    # out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    # depth               = cfg.MODEL.RESNETS.DEPTH
    # num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    # width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    # bottleneck_channels = num_groups * width_per_group
    # in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    # out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    # stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    # res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    # deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    # deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    # deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # # fmt: on
    # assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)
    #
    # num_blocks_per_stage = {
    #     18: [2, 2, 2, 2],
    #     34: [3, 4, 6, 3],
    #     50: [3, 4, 6, 3],
    #     101: [3, 4, 23, 3],
    #     152: [3, 8, 36, 3],
    # }[depth]
    #
    # if depth in [18, 34]:
    #     assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
    #     assert not any(
    #         deform_on_per_stage
    #     ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
    #     assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
    #     assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"
    #
    # stages = []
    #
    # # Avoid creating variables without gradients
    # # It consumes extra memory and may cause allreduce to fail
    # out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    # max_stage_idx = max(out_stage_idx)
    # for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
    #     dilation = res5_dilation if stage_idx == 5 else 1
    #     first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
    #     stage_kargs = {
    #         "num_blocks": num_blocks_per_stage[idx],
    #         "first_stride": first_stride,
    #         "in_channels": in_channels,
    #         "out_channels": out_channels,
    #         "norm": norm,
    #     }
    #     # Use BasicBlock for R18 and R34.
    #     if depth in [18, 34]:
    #         stage_kargs["block_class"] = BasicBlock
    #     else:
    #         stage_kargs["bottleneck_channels"] = bottleneck_channels
    #         stage_kargs["stride_in_1x1"] = stride_in_1x1
    #         stage_kargs["dilation"] = dilation
    #         stage_kargs["num_groups"] = num_groups
    #         if deform_on_per_stage[idx]:
    #             stage_kargs["block_class"] = DeformBottleneckBlock
    #             stage_kargs["deform_modulated"] = deform_modulated
    #             stage_kargs["deform_num_groups"] = deform_num_groups
    #         else:
    #             stage_kargs["block_class"] = BottleneckBlock
    #     blocks = make_stage(**stage_kargs)
    #     in_channels = out_channels
    #     out_channels *= 2
    #     bottleneck_channels *= 2
    #     stages.append(blocks)
    # return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)

# # ---------------------------------------------------------------------------- #
# # ResNet Conv Body
# # ---------------------------------------------------------------------------- #
# @registry.BACKBONES.register("resnet")
# def resnet(cfg):
#     model = ResNet(cfg)
#     if cfg.BACKBONE.RESNET.USE_WS:
#         model = convert_conv2convws_model(model)
#     return model
#
#
# @registry.BACKBONES.register("resnet_c4")
# def resnet(cfg):
#     model = ResNet(cfg, stride=16)
#     if cfg.BACKBONE.RESNET.USE_WS:
#         model = convert_conv2convws_model(model)
#     return model
#
#
# # ---------------------------------------------------------------------------- #
# # ResNet C5 Head
# # ---------------------------------------------------------------------------- #
# @registry.ROI_BOX_HEADS.register("resnet_c5_head")
# def resnet_c5_head(cfg, dim_in, spatial_scale):
#     model = ResNet_C5_Head(cfg, dim_in, spatial_scale)
#     if cfg.BACKBONE.RESNET.USE_WS:
#         model = convert_conv2convws_model(model)
#     return model
#
#
# # ---------------------------------------------------------------------------- #
# # ResNet 2mlp Head
# # ---------------------------------------------------------------------------- #
# @registry.ROI_BOX_HEADS.register("resnet_2mlp_head")
# def resnet_2mlp_head(cfg, dim_in, spatial_scale):
#     model = ResNet_2mlp_Head(cfg, dim_in, spatial_scale)
#     if cfg.BACKBONE.RESNET.USE_WS:
#         model = convert_conv2convws_model(model)
#     return model
