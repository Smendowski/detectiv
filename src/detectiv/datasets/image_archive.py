import os
import shutil
import stat
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from detectiv.data import TemporalSplit
from detectiv.datasets.image_folder import (
    ImageFolderReader,
    ImageFolderWriter,
    ImageFormat,
    ImageOutputConfig,
)
from detectiv.datasets.images import ImageDataset


class ImageArchiveWriter:
    def __init__(self, path: Path, image_format: ImageFormat = ImageFormat.PNG) -> None:
        if path.suffix != ".zip":
            raise ValueError("image archive path must end with .zip")
        self.path = path
        self.image_format = image_format

    def write(self, images: TemporalSplit[ImageDataset]) -> Path:
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


class ImageArtifactReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def open(self) -> Iterator[TemporalSplit[ImageDataset]]:
        if self.path.is_dir():
            yield ImageFolderReader(self.path).read()
            return
        if not zipfile.is_zipfile(self.path):
            raise ValueError("image artifact must be a directory or ZIP archive")
        with TemporaryDirectory(prefix="detectiv-artifact-") as name:
            root = Path(name)
            self._extract(root)
            yield ImageFolderReader(root).read()

    def _extract(self, root: Path) -> None:
        root = root.resolve()
        with zipfile.ZipFile(self.path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if member.flag_bits & 1:
                    raise ValueError("encrypted image archives are not supported")
                if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK:
                    raise ValueError("image archives must not contain symbolic links")
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError(
                        f"archive member escapes its root: {member.filename}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
