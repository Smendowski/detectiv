# Image Artifacts

::: detectiv.images.io

`ImageFolderWriter` writes a portable directory containing `artifact.json`, an
ordered `manifest.csv`, and NPY or PNG images. `ImageArchiveWriter` writes the
same layout as a ZIP archive. Use `ImageFolderReader` or `ImageArtifactReader`
to restore a `TemporalSplit[ImageDataset]`.

NPY artifacts preserve array values but reject object arrays so artifacts remain
safe to load with NumPy pickling disabled. PNG artifacts require finite RGB
values in the range `[0, 1]`.

Labelled datasets add non-pickled NPY sidecars under `point_labels/`, referenced
by split and series ID from `artifact.json`. Both folder and ZIP readers restore
the arrays as immutable `ImageDataset.point_labels`; metadata without those
references remains supported.
