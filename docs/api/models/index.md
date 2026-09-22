# Models Overview

Models operate on channel-first TS2I images. The current model family is an
autoencoder: an encoder maps images to embeddings, an optional bottleneck
transforms those embeddings, and a decoder reconstructs the original image.

`AutoencoderTrainer` fits a model on explicitly selected image indices. Its
configuration controls optimization, scheduling, validation, early stopping,
transfer learning, data loading, and runtime device selection. Training returns
an immutable `TrainingHistory` used by scenarios and tracking callbacks.

- [Autoencoders](autoencoders/index.md)
- [Autoencoder training](autoencoders/training.md)
- [Transfer learning](autoencoders/transfer-learning.md)
- [Runtime](runtime.md)
