# Materialization

Materialization renders lazy TS2I datasets to NPY files. It chooses or uses the
requested worker count, writes a manifest, validates the output, and publishes
the artifact only after a successful render. The returned
`MaterializedImageSplit` uses `NpyImageSource`, carries source-neutral
provenance and the typed materialization report, and can be passed directly to
a scenario.

`DataLoaderSettings` describes later PyTorch data-loader behavior and is stored
in the `MaterializationReport`; it does not control the rendering workers.

::: detectiv.ts2i.materialization
