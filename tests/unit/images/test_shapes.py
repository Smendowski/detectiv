from detectiv.images import ImageShape, ImageSize


def test_image_size_is_an_ordered_height_width_tuple() -> None:
    size = ImageSize(height=3, width=5)

    height, width = size

    assert size.height == 3
    assert size.width == 5
    assert (height, width) == (3, 5)
    assert tuple(size) == (3, 5)


def test_image_shape_exposes_its_spatial_size() -> None:
    shape = ImageShape(channels=1, height=3, width=5)

    assert shape.size == ImageSize(height=3, width=5)
    assert shape.shape == (1, 3, 5)
