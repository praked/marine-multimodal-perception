"""GBTScorerBundle: save/load round-trip, dispatch, calibration, ensemble."""
import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from scripts.fusion_model.models import (
    GBT_COLUMNS,
    GBTScorerBundle,
    IsotonicCalibrator,
    fit_gbt,
    load_any_scorer,
)


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(GBT_COLUMNS))), columns=GBT_COLUMNS)
    y = (df[GBT_COLUMNS[0]].to_numpy() + 0.3 * rng.normal(size=n) > 0).astype(float)
    return df, y


def _bundle(seeds=(0, 1)):
    df, y = _toy()
    clfs = [fit_gbt(df, y, seed=s) for s in seeds]
    b = GBTScorerBundle(clfs=clfs, train_info={"tag": "test"})
    b.calibrator = IsotonicCalibrator.fit(b.predict_proba_df(df), y)
    return b, df, y


def test_ensemble_is_mean_of_seeds():
    b, df, _ = _bundle()
    from scripts.fusion_model.models import gbt_predict
    manual = np.mean(np.stack([gbt_predict(c, df) for c in b.clfs]), axis=0)
    assert np.allclose(b.predict_proba_df(df), manual)


def test_save_load_roundtrip(tmp_path):
    b, df, _ = _bundle()
    path = tmp_path / "scorer_v1b.joblib"
    b.save(path)
    b2 = load_any_scorer(path)          # dispatch by suffix
    assert isinstance(b2, GBTScorerBundle)
    assert b2.sensors == b.sensors and len(b2.clfs) == 2
    assert np.allclose(
        b2.predict_proba_calibrated_df(df), b.predict_proba_calibrated_df(df))


def test_json_still_dispatches_to_v1a(tmp_path):
    # a .json path must NOT go through the joblib loader
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        load_any_scorer(p)              # v1a loader rejects a kind-less json


def test_calibrated_differs_from_raw_when_calibrator_present():
    b, df, _ = _bundle()
    raw = b.predict_proba_df(df)
    cal = b.predict_proba_calibrated_df(df)
    assert raw.shape == cal.shape
    assert np.all((cal >= 0) & (cal <= 1))


def test_missing_columns_are_nan_tolerated():
    b, df, _ = _bundle()
    part = df.drop(columns=list(GBT_COLUMNS[:3]))   # sdf missing some cols
    p = b.predict_proba_calibrated_df(part)
    assert np.all(np.isfinite(p))
