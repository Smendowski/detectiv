# Image Datasets

::: detectiv.images.types

`ImageSize` stores render dimensions in `(height, width)` order. `ImageShape`
adds the channel count and exposes channel-first `(channels, height, width)`
dimensions through its `shape` property.

::: detectiv.images.dataset

`ImageDataset` keeps image loading lazy while preserving each image's original
time-series `WindowReference`, partition-local `series_lengths`, and immutable
per-series `point_labels` when the source is labelled. Point labels describe
source time points and are not the derived `window_labels`. An `ImageSource`
supplies channel-first arrays with the dataset's declared `ImageShape`.

`ImageSplit` extends temporal train, validation, and test partitions with
source-neutral provenance. Materialized, folder, and ZIP-backed inputs use this
contract so scenario reports retain where their images came from.

::: detectiv.images.split
