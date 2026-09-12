from collections.abc import Callable, Sequence
from typing import cast

from torch import Tensor, nn

from detectiv.models.autoencoders.base import ImageDecoder
from detectiv.models.autoencoders.layers import make_activation, resolve_layer_values


class CNNDecoder(ImageDecoder):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        output_channels: int,
        *,
        kernel_sizes: int | Sequence[int] = 3,
        strides: int | Sequence[int] | None = None,
        output_kernel_size: int = 3,
        output_stride: int = 2,
        use_batch_norm: bool = True,
        activation: Callable[[], nn.Module] = nn.ReLU,
        output_activation: Callable[[], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if (
            input_channels <= 0
            or output_channels <= 0
            or output_kernel_size <= 0
            or output_stride <= 0
        ):
            raise ValueError(
                "channel counts, output_kernel_size, and output_stride must be positive"
            )
        if any(channels <= 0 for channels in hidden_channels):
            raise ValueError("hidden_channels must contain positive values")

        layer_strides = resolve_layer_values(
            strides, len(hidden_channels), name="strides", default=2
        )
        layer_kernel_sizes = resolve_layer_values(
            kernel_sizes, len(hidden_channels), name="kernel_sizes", default=3
        )
        layers: list[nn.Module] = []
        previous_channels = input_channels
        for next_channels, kernel_size, stride in zip(
            hidden_channels, layer_kernel_sizes, layer_strides, strict=True
        ):
            layers.append(
                nn.ConvTranspose2d(
                    previous_channels,
                    next_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    output_padding=stride - 1,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(next_channels))
            layers.append(make_activation(activation))
            previous_channels = next_channels
        layers.append(
            nn.ConvTranspose2d(
                previous_channels,
                output_channels,
                kernel_size=output_kernel_size,
                stride=output_stride,
                padding=output_kernel_size // 2,
                output_padding=output_stride - 1,
            )
        )
        if output_activation is not None:
            layers.append(make_activation(output_activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, embeddings: Tensor) -> Tensor:
        return cast(Tensor, self.layers(embeddings))
