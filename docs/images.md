# Images

The images domain bridges TS2I output and reconstruction models without losing
the original time-series window identity.

`ImageDataset` is lazy. It keeps a channel-first `ImageShape`, ordered
`WindowReference` values, optional binary window labels, an explicit series ID
and scalar series length, and a
source that materializes an image only when indexed. For labelled sources it
also keeps one immutable, partition-local `point_labels` array. Point labels
align with `series_length` and remain separate from labels derived for
image windows; unlabelled datasets retain `None`. `TorchImageDataset` adapts
that same data to a float32 PyTorch dataset, optionally with an ordered subset.

## Artifacts

Folder and ZIP artifacts contain an `artifact.json` descriptor, ordered
`manifest.csv`, and images partitioned by split and label. Readers restore a
`TemporalSplit[ImageDataset]` without eagerly loading image arrays.

When point labels are present, writers store each split as a non-pickled NPY
sidecar under `point_labels/` and record its path in
`artifact.json`. Folder and ZIP readers validate and restore those sidecars.
The pre-release artifact schema intentionally does not support old multi-series
artifacts.

NPY is the default format and preserves numeric arrays. It rejects object arrays
so NumPy can reload artifacts with pickling disabled. PNG is intended for RGB
images and requires finite values in `[0, 1]`.

Use the API reference for exact contracts:

- [Image datasets](api/images/dataset.md)
- [Torch adapter](api/images/torch.md)
- [Image artifacts](api/images/io.md)
