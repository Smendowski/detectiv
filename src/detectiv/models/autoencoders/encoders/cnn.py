from collections.abc import Callable, Sequence
from typing import cast

from torch import Tensor, nn

from detectiv.models.autoencoders.base import ImageEncoder
from detectiv.models.autoencoders.layers import make_activation, resolve_layer_values


class CNNEncoder(ImageEncoder):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        *,
        kernel_sizes: int | Sequence[int] = 3,
        strides: int | Sequence[int] | None = None,
        use_batch_norm: bool = True,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if not hidden_channels or any(channels <= 0 for channels in hidden_channels):
            raise ValueError("hidden_channels must contain positive values")

        layer_strides = resolve_layer_values(
            strides, len(hidden_channels), name="strides", default=2
        )
        layer_kernel_sizes = resolve_layer_values(
            kernel_sizes, len(hidden_channels), name="kernel_sizes", default=3
        )
        layers: list[nn.Module] = []
        previous_channels = input_channels
        for output_channels, kernel_size, stride in zip(
            hidden_channels, layer_kernel_sizes, layer_strides, strict=True
        ):
            layers.append(
                nn.Conv2d(
                    previous_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(output_channels))
            layers.append(make_activation(activation))
            previous_channels = output_channels
        self.layers = nn.Sequential(*layers)
        self.output_channels = previous_channels

    def forward(self, images: Tensor) -> Tensor:
        return cast(Tensor, self.layers(images))
