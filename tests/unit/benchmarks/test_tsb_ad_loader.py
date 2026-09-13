from pathlib import Path

import numpy as np
import pytest

from detectiv.benchmarks.tsb_ad import (
    TSBADCollection,
    TSBADCollectionLoader,
    TSBADCsvLoader,
)


def test_csv_loader_preserves_feature_names_and_single_rows(tmp_path: Path) -> None:
    path = tmp_path / "001_NAB_id_1_Facility_tr_1_1st_2.csv"
    path.write_text("left,right,Label\n1.0,2.0,0\n", encoding="utf-8")

    series = TSBADCsvLoader().load(path)

    assert series.series_id == path.stem
    assert series.feature_names == ("left", "right")
    np.testing.assert_array_equal(series.values, [[1.0, 2.0]])
    np.testing.assert_array_equal(series.labels, [False])


def test_collection_loader_groups_series_and_returns_splitters(tmp_path: Path) -> None:
    collection = tmp_path / TSBADCollection.MULTIVARIATE
    collection.mkdir()
    _write_csv(
        collection / "001_SMD_id_1_Facility_tr_2_1st_3.csv",
        "first,second,Label\n1,2,0\n3,4,0\n5,6,1\n7,8,1\n",
    )
    _write_csv(
        collection / "002_SMD_id_2_Facility_tr_3_1st_4.csv",
        "first,second,Label\n1,2,0\n3,4,0\n5,6,0\n7,8,1\n9,10,1\n",
    )
    _write_csv(
        collection / "003_MSL_id_1_Sensor_tr_2_1st_3.csv",
        "first,second,Label\n1,2,0\n3,4,0\n5,6,1\n7,8,1\n",
    )

    loader = TSBADCollectionLoader(tmp_path)

    datasets = loader.load_collection(TSBADCollection.MULTIVARIATE)

    assert loader.dataset_names(TSBADCollection.MULTIVARIATE) == ("SMD", "MSL")
    assert [item.dataset.dataset_id for item in datasets] == [
        "TSB-AD-M:SMD",
        "TSB-AD-M:MSL",
    ]
    smd = datasets[0]
    assert len(smd.dataset) == 2
    assert smd.dataset.metadata["collection"] == "TSB-AD-M"
    split = smd.dataset.split(smd.splitter)
    assert split.train["001_SMD_id_1_Facility_tr_2_1st_3"].n_timesteps == 2
    assert split.test["002_SMD_id_2_Facility_tr_3_1st_4"].n_timesteps == 2
    msl = loader.load_dataset(TSBADCollection.MULTIVARIATE, "MSL")
    assert msl.dataset.series_ids == ("003_MSL_id_1_Sensor_tr_2_1st_3",)


def test_collection_loader_rejects_invalid_benchmark_files(tmp_path: Path) -> None:
    collection = tmp_path / TSBADCollection.UNIVARIATE
    collection.mkdir()
    _write_csv(collection / "series.csv", "value,Label\n1,0\n2,1\n")

    with pytest.raises(ValueError, match="invalid TSB-AD filename"):
        TSBADCollectionLoader(tmp_path).load_collection(TSBADCollection.UNIVARIATE)


def test_collection_loader_rejects_train_boundaries_outside_series(
    tmp_path: Path,
) -> None:
    collection = tmp_path / TSBADCollection.UNIVARIATE
    collection.mkdir()
    _write_csv(
        collection / "001_NAB_id_1_Facility_tr_2_1st_2.csv",
        "value,Label\n1,0\n2,1\n",
    )

    with pytest.raises(ValueError, match="train boundary"):
        TSBADCollectionLoader(tmp_path).load_collection(TSBADCollection.UNIVARIATE)


def test_collection_loader_validates_collection_dimensionality(tmp_path: Path) -> None:
    collection = tmp_path / TSBADCollection.UNIVARIATE
    collection.mkdir()
    _write_csv(
        collection / "001_NAB_id_1_Facility_tr_1_1st_2.csv",
        "left,right,Label\n1,2,0\n3,4,1\n",
    )

    with pytest.raises(ValueError, match="must be univariate"):
        TSBADCollectionLoader(tmp_path).load_collection(TSBADCollection.UNIVARIATE)


def test_collection_loader_rejects_unknown_dataset_names(tmp_path: Path) -> None:
    collection = tmp_path / TSBADCollection.UNIVARIATE
    collection.mkdir()
    _write_csv(
        collection / "001_NAB_id_1_Facility_tr_1_1st_2.csv",
        "value,Label\n1,0\n2,1\n",
    )

    with pytest.raises(ValueError, match="does not contain dataset"):
        TSBADCollectionLoader(tmp_path).load_dataset(TSBADCollection.UNIVARIATE, "SMD")


def _write_csv(path: Path, values: str) -> None:
    path.write_text(values, encoding="utf-8")
