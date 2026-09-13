import shutil
import stat
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from detectiv.images import ImageDataset
from detectiv.images.io.readers.images import ImageFolderReader
from detectiv.time_series import TemporalSplit


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
