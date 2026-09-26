from pathlib import Path

import numpy as np
import pytest

from detectiv.benchmarks.tsb_ad import (
    TSBADCollection,
    TSBADCollectionLoader,
    load_tsb_ad_csv,
)


def test_csv_loader_preserves_feature_names_and_single_rows(tmp_path: Path) -> None:
    path = tmp_path / "001_NAB_id_1_Facility_tr_1_1st_2.csv"
    path.write_text("left,right,Label\n1.0,2.0,0\n", encoding="utf-8")

    series = load_tsb_ad_csv(path)

    assert series.series_id == path.stem
    assert series.feature_names == ("left", "right")
    np.testing.assert_array_equal(series.values, [[1.0, 2.0]])
    np.testing.assert_array_equal(series.labels, [False])


def test_collection_yields_each_csv_as_an_independent_split(tmp_path: Path) -> None:
    collection = tmp_path / TSBADCollection.MULTIVARIATE
    collection.mkdir()
    first = collection / "001_SMD_id_1_Facility_tr_2_1st_3.csv"
    second = collection / "002_SMD_id_2_Facility_tr_3_1st_4.csv"
    third = collection / "003_MSL_id_1_Sensor_tr_2_1st_3.csv"
    _write_csv(first, "first,second,Label\n1,2,0\n3,4,0\n5,6,1\n7,8,1\n")
    _write_csv(
        second,
        "first,second,Label\n1,2,0\n3,4,0\n5,6,0\n7,8,1\n9,10,1\n",
    )
    _write_csv(third, "first,second,Label\n1,2,0\n3,4,0\n5,6,1\n7,8,1\n")

    loader = TSBADCollectionLoader(tmp_path)
    splits = tuple(loader.iter_collection("TSB-AD-M"))

    assert loader.source_groups(TSBADCollection.MULTIVARIATE) == ("SMD", "MSL")
    assert [split.train.series_id for split in splits] == [
        first.stem,
        second.stem,
        third.stem,
    ]
    assert [split.train.n_timesteps for split in splits] == [2, 3, 2]
    assert [
        split.train.series_id
        for split in loader.iter_collection("TSB-AD-M", source_group="MSL")
    ] == [third.stem]
    loaded = loader.load_series(TSBADCollection.MULTIVARIATE, third.stem)
    assert loaded.train.series_id == third.stem
    assert loaded.train.n_timesteps == 2


def test_collection_loader_is_lazy(tmp_path: Path) -> None:
    collection = tmp_path / TSBADCollection.MULTIVARIATE
    collection.mkdir()
    first = collection / "001_SMD_id_1_Facility_tr_1_1st_2.csv"
    second = collection / "002_MSL_id_1_Sensor_tr_1_1st_2.csv"
    _write_csv(first, "first,second,Label\n1,2,0\n3,4,1\n")
    _write_csv(second, "first,second,Label\ninvalid,row,values\n")
    loader = TSBADCollectionLoader(tmp_path)
    splits = loader.iter_collection(TSBADCollection.MULTIVARIATE)

    assert next(splits).train.series_id == first.stem
    with pytest.raises(ValueError):
        next(splits)


@pytest.mark.parametrize(
    ("filename", "values", "message"),
    [
        ("series.csv", "value,Label\n1,0\n2,1\n", "invalid TSB-AD filename"),
        (
            "001_NAB_id_1_Facility_tr_2_1st_2.csv",
            "value,Label\n1,0\n2,1\n",
            "train_end",
        ),
        (
            "001_NAB_id_1_Facility_tr_1_1st_2.csv",
            "left,right,Label\n1,2,0\n3,4,1\n",
            "must be univariate",
        ),
    ],
)
def test_collection_loader_rejects_invalid_series(
    tmp_path: Path, filename: str, values: str, message: str
) -> None:
    collection = tmp_path / TSBADCollection.UNIVARIATE
    collection.mkdir()
    _write_csv(collection / filename, values)

    with pytest.raises(ValueError, match=message):
        next(TSBADCollectionLoader(tmp_path).iter_collection("TSB-AD-U"))


def test_collection_loader_rejects_unknown_series(tmp_path: Path) -> None:
    collection = tmp_path / TSBADCollection.UNIVARIATE
    collection.mkdir()
    _write_csv(
        collection / "001_NAB_id_1_Facility_tr_1_1st_2.csv",
        "value,Label\n1,0\n2,1\n",
    )

    with pytest.raises(ValueError, match="does not contain series"):
        TSBADCollectionLoader(tmp_path).load_series("TSB-AD-U", "missing")


def test_csv_loader_preserves_filesystem_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tsb_ad_csv(tmp_path / "missing.csv")


def _write_csv(path: Path, values: str) -> None:
    path.write_text(values, encoding="utf-8")
