import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from detectiv.images import ImageDataset
from detectiv.images.io import ImageFormat, ImageOutputConfig
from detectiv.images.io.writers.images import ImageFolderWriter
from detectiv.time_series import TemporalSplit


class ImageArchiveWriter:
    """Write a temporal image split as a ZIP artifact."""

    def __init__(self, path: Path, image_format: ImageFormat = ImageFormat.NPY) -> None:
        if path.suffix != ".zip":
            raise ValueError("image archive path must end with .zip")
        if not isinstance(image_format, ImageFormat):
            raise TypeError("image_format must be an ImageFormat")
        self.path = path
        self.image_format = image_format

    def write(self, images: TemporalSplit[ImageDataset]) -> Path:
        """Write image splits and point-label sidecars to a ZIP artifact.

        Args:
            images: Temporal image split to serialize.

        Returns:
            Path to the completed ZIP artifact.

        Raises:
            FileExistsError: If the destination already exists.
        """
        if self.path.exists():
            raise FileExistsError(f"output already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=f".{self.path.stem}.", dir=self.path.parent
        ) as name:
            temporary = Path(name)
            folder = ImageFolderWriter(
                ImageOutputConfig(temporary / "artifact", format=self.image_format)
            ).write(images)
            archive = temporary / self.path.name
            with zipfile.ZipFile(
                archive,
                "x",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as output:
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        output.write(path, path.relative_to(folder))
            os.replace(archive, self.path)
        return self.path
