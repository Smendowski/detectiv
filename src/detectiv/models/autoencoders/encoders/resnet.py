from typing import cast

import torch
from torch import Tensor, nn

from detectiv.models.autoencoders.base import ImageEncoder


class ResNet18Encoder(ImageEncoder):
    def __init__(
        self,
        *,
        input_channels: int = 3,
        pretrained: bool = True,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise ImportError(
                "ResNet18Encoder requires the vision extra: uv add 'detectiv[vision]'"
            ) from error
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        if input_channels != backbone.conv1.in_channels:
            backbone.conv1 = _input_convolution(
                backbone.conv1,
                input_channels,
                pretrained=pretrained,
            )
        self.layers = nn.Sequential(*tuple(backbone.children())[:-2])
        self.output_channels = 512
        if frozen:
            self.freeze()

    def forward(self, images: Tensor) -> Tensor:
        return cast(Tensor, self.layers(images))


def _input_convolution(
    convolution: nn.Conv2d,
    input_channels: int,
    *,
    pretrained: bool,
) -> nn.Conv2d:
    adapted = nn.Conv2d(
        input_channels,
        convolution.out_channels,
        cast(tuple[int, int], convolution.kernel_size),
        stride=cast(tuple[int, int], convolution.stride),
        padding=cast(str | tuple[int, int], convolution.padding),
        dilation=cast(tuple[int, int], convolution.dilation),
        groups=convolution.groups,
        bias=convolution.bias is not None,
        padding_mode=convolution.padding_mode,
    )
    if not pretrained:
        return adapted
    weights = convolution.weight.mean(dim=1, keepdim=True)
    weights = weights.repeat(1, input_channels, 1, 1)
    with torch.no_grad():
        adapted.weight.copy_(weights * (convolution.in_channels / input_channels))
        if convolution.bias is not None and adapted.bias is not None:
            adapted.bias.copy_(convolution.bias)
    return adapted
