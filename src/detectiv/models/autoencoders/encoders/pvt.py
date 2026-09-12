from typing import cast

from torch import Tensor

from detectiv.models.autoencoders.base import ImageEncoder


class PVTv2B1Encoder(ImageEncoder):
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
            from timm import create_model
        except ImportError as error:
            raise ImportError(
                "PVTv2B1Encoder requires the vision extra: uv add 'detectiv[vision]'"
            ) from error
        self.backbone = create_model(
            "pvt_v2_b1",
            pretrained=pretrained,
            features_only=True,
            in_chans=input_channels,
        )
        self.output_channels = 512
        if frozen:
            self.freeze()

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone(images)
        return cast(Tensor, features[-1])
