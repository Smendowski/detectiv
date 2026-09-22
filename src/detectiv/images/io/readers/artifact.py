import shutil
import stat
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from detectiv.images import ImageDataset, ImageSplit
from detectiv.images.io.readers.images import ImageFolderReader
from detectiv.time_series import TemporalSplit

MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1 << 30


class ImageArtifactReader:
    """Restore a temporal image split from a folder or guarded ZIP artifact."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def open(self) -> Iterator[ImageSplit]:
        """Open a folder or guarded ZIP artifact as a lazy image split.

        Returns:
            An iterator context yielding the restored image split with source
            provenance.

        Raises:
            ValueError: If the path or ZIP contents are invalid.
        """
        if self.path.is_dir():
            yield self._with_provenance(ImageFolderReader(self.path).read(), "folder")
            return
        if not zipfile.is_zipfile(self.path):
            raise ValueError("image artifact must be a directory or ZIP archive")
        with TemporaryDirectory(prefix="detectiv-artifact-") as name:
            root = Path(name)
            self._extract(root)
            yield self._with_provenance(ImageFolderReader(root).read(), "zip")

    def _with_provenance(
        self, images: TemporalSplit[ImageDataset], source: str
    ) -> ImageSplit:
        return ImageSplit(
            train=images.train,
            validation=images.validation,
            test=images.test,
            provenance={"source": source, "location": str(self.path)},
        )

    def _extract(self, root: Path) -> None:
        root = root.resolve()
        with zipfile.ZipFile(self.path) as archive:
            members = tuple(
                member for member in archive.infolist() if not member.is_dir()
            )
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("image archive contains too many members")
            if (
                sum(member.file_size for member in members)
                > MAX_ARCHIVE_UNCOMPRESSED_BYTES
            ):
                raise ValueError("image archive expands beyond the allowed size")
            for member in members:
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
