import zipfile
from pathlib import Path

import pytest

from detectiv.images.io.readers import ImageArtifactReader
from detectiv.images.io.readers import artifact as artifact_module


def test_artifact_reader_rejects_archives_larger_than_the_allowed_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("image.npy", b"image")
    monkeypatch.setattr(artifact_module, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", 0)

    with pytest.raises(ValueError, match="allowed size"):
        ImageArtifactReader(archive)._extract(tmp_path / "extracted")
