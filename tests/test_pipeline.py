from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from pipeline import (Config, FEATURES, features, inference_inputs, label_outcome,
                      label_windows, load_dataset, load_metadata, make_dataset,
                      prepare, read_prices, validate_prices)


def test_csv_header_and_validation(csv_file, prices):
    result = read_prices(csv_file)
    assert len(result) == len(prices)
    assert result.datetime.iloc[0] == prices.datetime.iloc[0]
    with pytest.raises(ValueError, match='duplicate'):
        validate_prices(pd.concat([prices, prices.iloc[:1]]))
    invalid = prices.copy()
    invalid.loc[0, 'close'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        validate_prices(invalid)
    invalid.loc[0, 'close'] = 5
    with pytest.raises(ValueError, match='bounds'):
        validate_prices(invalid)


@pytest.mark.parametrize('candles,expected', [
    ([(101, 99.5), (103, 99.5)], 0),
    ([(100.5, 99), (100.5, 97)], 1),
    ([(103, 98)], 2),  # both barriers in one candle: stop wins
    ([(101, 99)], 2),  # equality stops both directions
    ([(102, 99.5)], 0),  # target equality counts
    ([(100.5, 98)], 1),
    ([(100.5, 99.5)], 2),
    ([(101, 99.5), (103, 99.5)], 0),
])
def test_labels(candles, expected):
    assert label_outcome(100, pd.DataFrame(candles, columns=['high', 'low']), 1, 2) == expected


def test_ambiguous_dual_winners():
    future = pd.DataFrame([(101, 100), (100, 99)], columns=['high', 'low'])
    assert label_outcome(100, future, 2, 1) == 2


@pytest.mark.parametrize('index', [[100, 200, 300, 400], pd.date_range('2020-01-01', periods=4)])
def test_window_alignment_and_entry_candle_exclusion(index):
    price = pd.DataFrame(dict(close=[100] * 4, high=[110, 110, 100.5, 100.5],
                              low=[90, 90, 99.5, 99.5], marker=[0, 1, 2, 3]), index=index)
    windows, labels = label_windows(price, 1, 2, 2, 2, ['marker'])
    assert len(windows) == 1
    assert windows[0].ravel().tolist() == [0, 1]
    assert labels == [2]
    from AssistanceFunctions import DeepLearning
    actual, outcome = DeepLearning.label_creator(sl=1, tp=2, price=price, look_forward=2,
                                                 look_back=2, columns_wanted=['marker'])
    np.testing.assert_array_equal(actual, windows)
    assert outcome == labels


def test_prepare_scaler_and_inference_parity(csv_file, prices, tmp_path):
    output = tmp_path / 'bundle'
    counts = prepare(csv_file, output)
    config, scaler = load_metadata(output)
    train_frame = features(read_prices(csv_file).iloc[:int(len(prices) * .72)])
    expected = MinMaxScaler(feature_range=(-1, 1)).fit(train_frame[FEATURES])
    np.testing.assert_allclose(scaler.data_max_, expected.data_max_)
    datasets = [load_dataset(output, name, config) for name in ['train', 'validation', 'test']]
    assert all(count > 0 for count in counts.values())
    for left, right in zip(datasets, datasets[1:]):
        assert left['timestamps'][-1] < right['timestamps'][0]
    # Reconstruct the exact history available at the first sample's entry.
    endpoint = pd.Timestamp(datasets[0]['timestamps'][0])
    inputs = inference_inputs(read_prices(csv_file).loc[lambda x: x.datetime <= endpoint], scaler, config)
    for actual, key in zip(inputs.values(), ['ta', 'ohlc']):
        np.testing.assert_allclose(actual[0], datasets[0][key][0], atol=1e-6)
    # Distribution changes in future partitions must not change the fitted scaler.
    assert scaler.n_samples_seen_ == len(train_frame)
    with pytest.raises(ValueError, match='empty'):
        prepare(csv_file, output)
    metadata = json.loads((output / 'metadata.json').read_text())
    metadata['features'].reverse()
    (output / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Incompatible'):
        load_metadata(output)


def test_shapes_and_invalid_inputs(prices):
    config = Config()
    frame = features(prices.iloc[:300])
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(frame[FEATURES])
    dataset = make_dataset(frame, scaler, config)
    assert dataset['ta'].shape == (len(frame) - 44, 30, 4)
    assert dataset['ohlc'].shape == dataset['ta'].shape
    with pytest.raises(ValueError, match='Insufficient'):
        inference_inputs(prices.iloc[:200], scaler, config)
    with pytest.raises(ValueError):
        replace(config, sl=0)


def test_imports_have_no_execution_side_effects():
    command = "import FeatureCreation, model, MetaTrader, OrderAutomation, AssistanceFunctions, main; import sys; assert 'MetaTrader5' not in sys.modules; assert 'tensorflow' not in sys.modules"
    subprocess.run([sys.executable, '-c', command], check=True, timeout=30)
    result = subprocess.run([sys.executable, 'OrderAutomation.py'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'disabled' in result.stderr
    assert Path('main.py').read_bytes() == Path('AssistanceFunctions.py').read_bytes()


def test_decimal_barrier_equality():
    future = pd.DataFrame([(0.3, 0.1)], columns=['high', 'low'])
    assert label_outcome(0.1, future, sl=0.05, tp=0.2) == 0
