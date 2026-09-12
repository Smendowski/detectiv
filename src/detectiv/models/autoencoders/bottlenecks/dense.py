from collections.abc import Callable
from typing import cast

from torch import Tensor, nn

from detectiv.models.autoencoders.base import ImageBottleneck
from detectiv.models.autoencoders.layers import make_activation


class DenseSpatialBottleneck(ImageBottleneck):
    def __init__(
        self,
        input_channels: int,
        input_size: tuple[int, int],
        latent_dim: int,
        *,
        adaptive_pool: bool = False,
        activation: Callable[[], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if (
            input_channels <= 0
            or latent_dim <= 0
            or any(size <= 0 for size in input_size)
        ):
            raise ValueError(
                "channels, spatial dimensions, and latent_dim must be positive"
            )
        self.input_size = input_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1) if adaptive_pool else None
        encoded_size = (1, 1) if adaptive_pool else input_size
        n_features = input_channels * encoded_size[0] * encoded_size[1]
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, latent_dim),
            *(() if activation is None else (make_activation(activation),)),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, input_channels * input_size[0] * input_size[1]),
            *(() if activation is None else (make_activation(activation),)),
            nn.Unflatten(1, (input_channels, *input_size)),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        if self.adaptive_pool is not None:
            embeddings = self.adaptive_pool(embeddings)
        return cast(Tensor, self.from_latent(self.to_latent(embeddings)))
