# Runs API

`RunArtifactWriter` and `RunArtifactCallback` publish local run bundles. A
completed report records resolved scenario inputs, reproducibility settings,
training device, runtime details, source revision and dirty state, and the
dependency-lock checksum. Use `RunArtifacts.open_verified()` for data that must
pass checksum verification before it is read.
