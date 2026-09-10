import argparse
from pathlib import Path

from loguru import logger

from detectiv import (
    FixedProjectionStrategy,
    Identity,
    ImageFolderWriter,
    ImageFormat,
    ImageOutputConfig,
    ImagePreparation,
    ProjectionScheme,
    RandomNoise,
    TailPolicy,
    TemporalBoundary,
    TemporalSplitter,
    TimeSeriesDataset,
    TSBADCsvLoader,
    WindowLabelingStrategy,
    WindowSpec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", type=int, required=True)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output-format",
        type=ImageFormat,
        choices=list(ImageFormat),
        default=ImageFormat.PNG,
    )
    parser.add_argument(
        "--labeling-strategy",
        type=WindowLabelingStrategy,
        choices=list(WindowLabelingStrategy),
        default=WindowLabelingStrategy.OR_POOLING,
    )
    args = parser.parse_args()

    series = TSBADCsvLoader().load(args.path)
    series_id = series.series_id
    if series_id is None:
        raise RuntimeError("loader must assign a series ID")
    dataset = TimeSeriesDataset(series_id, {series_id: series})
    projection = FixedProjectionStrategy(
        ProjectionScheme(Identity()).channels(RandomNoise()).replicate(n_channels=3)
    )
    images = (
        ImagePreparation(dataset)
        .split(
            TemporalSplitter({series_id: TemporalBoundary(train_end=args.train_end)})
        )
        .window(
            train=WindowSpec(
                args.length,
                labeling_strategy=args.labeling_strategy,
            ),
            test=WindowSpec(
                args.length,
                stride=1,
                tail=TailPolicy.EDGE_PAD,
                labeling_strategy=args.labeling_strategy,
            ),
        )
        .project(projection)
        .build((args.size, args.size), seed=args.seed)
    )
    output = ImageFolderWriter(
        ImageOutputConfig(
            args.output,
            args.output_format,
            workers=args.workers,
        )
    ).write(images)
    logger.info(
        "wrote {} train and {} test images to {}",
        len(images.train),
        len(images.test),
        output,
    )


if __name__ == "__main__":
    main()
