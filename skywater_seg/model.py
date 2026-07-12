"""
Model factory for sky/water segmentation.

Uses segmentation-models-pytorch for battle-tested architectures.
Supports DeepLabV3+, U-Net, FPN, PSPNet and more with various encoders.

Extended encoders (beyond SMP built-ins):
  - convnext-tiny  (timm convnext_tiny, 29M)
  - convnext-small (timm convnext_small, 50M)
  - convnext-base  (timm convnext_base, 89M)

DINOv3-distilled weights: use `encoder_weights: dinov3` to load Meta's
DINOv3-distilled ConvNeXt weights (e.g. convnext_tiny.dinov3_lvd1689m).
"""

from typing import Optional

import torch.nn as nn
from loguru import logger

from skywater_seg.config import Config

# ── Registry of extra timm encoders not built into SMP ──
_EXTRA_ENCODERS = {
    "convnext-tiny": {
        "timm_name": "convnext_tiny",
        "params": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    },
    "convnext-small": {
        "timm_name": "convnext_small",
        "params": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
    },
    "convnext-base": {
        "timm_name": "convnext_base",
        "params": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
    },
}



def create_model(config: Config) -> nn.Module:
    """Create a segmentation model from config.

    Args:
        config: Configuration object.  Supports:
          - All SMP built-in encoders (resnet, efficientnet, mit, etc.)
          - Extra encoders: convnext-tiny, convnext-small, convnext-base
          - encoder_weights: "imagenet" | "dinov3" | None | path
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation-models-pytorch is required. "
            "Install with: pip install segmentation-models-pytorch"
        )

    model_name = config.model.name.lower()
    encoder = config.model.encoder_name
    use_dinov3 = config.model.encoder_weights == "dinov3"

    # ── Handle ConvNeXt (not built into SMP) ──
    if encoder in _EXTRA_ENCODERS:
        return _create_convnext_model(config, smp)

    # ── SMP native encoders ──
    weights = config.model.encoder_weights if not use_dinov3 else None
    common_kwargs = dict(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=config.model.in_channels,
        classes=config.model.classes,
    )

    if model_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_output_stride=config.model.encoder_output_stride,
            decoder_channels=config.model.decoder_channels,
            decoder_atrous_rates=tuple(config.model.decoder_atrous_rates),
            **common_kwargs,
        )
    elif model_name == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_output_stride=config.model.encoder_output_stride,
            **common_kwargs,
        )
    elif model_name == "unet":
        model = smp.Unet(decoder_channels=(256, 128, 64, 32, 16), **common_kwargs)
    elif model_name == "unetplusplus":
        model = smp.UnetPlusPlus(decoder_channels=(256, 128, 64, 32, 16), **common_kwargs)
    elif model_name == "fpn":
        model = smp.FPN(**common_kwargs)
    elif model_name == "pspnet":
        model = smp.PSPNet(**common_kwargs)
    elif model_name == "pan":
        model = smp.PAN(**common_kwargs)
    elif model_name == "linknet":
        model = smp.Linknet(**common_kwargs)
    elif model_name == "segformer":
        model = smp.Segformer(**common_kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: deeplabv3plus, deeplabv3, unet, unetplusplus, "
            f"fpn, pspnet, pan, linknet, segformer"
        )

    return model


def _create_convnext_model(config: Config, smp) -> nn.Module:
    """Create DeepLabV3+ with ConvNeXt encoder.

    SMP doesn't include ConvNeXt natively, so we build encoder from timm
    and assemble the full model (encoder + decoder + head) manually.
    """
    import timm
    from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3PlusDecoder
    from segmentation_models_pytorch.base import SegmentationHead, initialization
    from segmentation_models_pytorch.encoders._base import EncoderMixin

    encoder_name = config.model.encoder_name
    info = _EXTRA_ENCODERS[encoder_name]
    timm_name = info["timm_name"]

    # Load ConvNeXt with requested weights
    if config.model.encoder_weights == "dinov3":
        timm_tag = f"{timm_name}.dinov3_lvd1689m"
        logger.info(f"[ConvNeXt] DINOv3 weights: {timm_tag}")
    elif config.model.encoder_weights == "imagenet":
        timm_tag = f"{timm_name}.fb_in22k_ft_in1k"  # Meta's best ImageNet weights
        logger.info(f"[ConvNeXt] ImageNet-22K weights: {timm_tag}")
    elif config.model.encoder_weights in (None, "random"):
        timm_tag = timm_name
        logger.info(f"[ConvNeXt] Random init: {timm_tag}")
    else:
        timm_tag = config.model.encoder_weights  # custom checkpoint tag
        logger.info(f"[ConvNeXt] Custom weights: {timm_tag}")

    timm_encoder = timm.create_model(timm_tag, features_only=True, pretrained=True)

    # ── Build SMP-compatible encoder wrapper ──
    # ConvNeXt has 4 stages (strides 4,8,16,32), but SMP DeepLabV3+ expects
    # 5 stages (strides 2,4,8,16,32). We prepend a lightweight stem conv
    # to create the missing stride-2 stage.
    class ConvNeXtEncoder(nn.Module, EncoderMixin):
        def __init__(self, timm_model, timm_channels):
            super().__init__()
            self._timm = timm_model
            # Add a stride-2 stem so we have 5 stages matching SMP's expectation
            self.stem = nn.Sequential(
                nn.Conv2d(3, timm_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(timm_channels[0]),
                nn.ReLU(inplace=True),
            )
            # Channels: [stem_out, stage0, stage1, stage2, stage3]
            self._out_channels = [timm_channels[0]] + list(timm_channels)
            self._depth = len(self._out_channels)
            self._in_channels = 3

        def forward(self, x):
            stem_feat = self.stem(x)               # stride 2
            timm_feats = self._timm(x)             # strides 4, 8, 16, 32
            return [stem_feat] + list(timm_feats)  # 5 features total

        @property
        def out_channels(self):
            return self._out_channels

    encoder = ConvNeXtEncoder(timm_encoder, timm_encoder.feature_info.channels())

    # ── Build decoder + segmentation head ──
    decoder = DeepLabV3PlusDecoder(
        encoder_channels=encoder.out_channels,
        encoder_depth=encoder._depth,
        out_channels=config.model.decoder_channels,
        atrous_rates=tuple(config.model.decoder_atrous_rates),
        output_stride=config.model.encoder_output_stride,
        aspp_separable=False,
        aspp_dropout=0.1,
    )

    head = SegmentationHead(
        in_channels=config.model.decoder_channels,
        out_channels=config.model.classes,
        kernel_size=1,
    )

    # ── Assemble full model ──
    model = _AssembledModel(encoder, decoder, head, config.model.classes,
                            output_stride=config.model.encoder_output_stride)
    initialization.initialize_decoder(decoder)
    initialization.initialize_head(head)

    return model


class _AssembledModel(nn.Module):
    """Complete segmentation model: encoder + decoder + head."""
    def __init__(self, encoder, decoder, head, num_classes, output_stride=16):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = head
        self.num_classes = num_classes
        # Decoder uses features[2] (stride 8) as skip. Output at stride 8.
        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        masks = self.upsample(masks)
        return masks


def get_model_info(model: nn.Module) -> dict:
    """Get model statistics: parameter count, size estimate, etc."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024 ** 2)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb_float32": round(size_mb, 2),
        "size_mb_fp16": round(size_mb / 2, 2),
    }


PRESETS = {
    "lightweight": {
        "name": "deeplabv3plus",
        "encoder_name": "timm-mobilenetv3_large_100",
        "encoder_weights": "imagenet",
        "classes": 4,
        "encoder_output_stride": 16,
        "decoder_channels": 256,
    },
    "ultra-lightweight": {
        "name": "deeplabv3plus",
        "encoder_name": "timm-mobilenetv3_small_050",
        "encoder_weights": "imagenet",
        "classes": 4,
        "encoder_output_stride": 16,
        "decoder_channels": 128,
    },
    "balanced": {
        "name": "deeplabv3plus",
        "encoder_name": "timm-efficientnet-b0",
        "encoder_weights": "imagenet",
        "classes": 4,
        "encoder_output_stride": 16,
        "decoder_channels": 256,
    },
    "accurate": {
        "name": "deeplabv3plus",
        "encoder_name": "timm-efficientnet-b3",
        "encoder_weights": "imagenet",
        "classes": 4,
        "encoder_output_stride": 16,
        "decoder_channels": 256,
    },
    "convnext_dinov3": {
        "name": "deeplabv3plus",
        "encoder_name": "convnext-tiny",
        "encoder_weights": "dinov3",
        "classes": 4,
        "encoder_output_stride": 16,
        "decoder_channels": 256,
        "decoder_atrous_rates": (12, 24, 36),
    },
}


def create_model_from_preset(preset_name: str) -> nn.Module:
    """Create model from a predefined preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}"
        )
    import segmentation_models_pytorch as smp
    preset = PRESETS[preset_name]
    return smp.DeepLabV3Plus(**preset)
