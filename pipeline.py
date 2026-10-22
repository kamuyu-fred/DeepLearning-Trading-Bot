"""Versioned, causal preprocessing shared by training and completed-bar inference."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

OHLC = ['open', 'high', 'low', 'close']
TECH = ['ema_mid', 'ema_long', 'ema_short', 'atr']
LOG = [f'log_{name}' for name in OHLC]
FEATURES = TECH + LOG
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    look_back: int = 30
    look_forward: int = 15
    sl: float = 0.0010
    tp: float = 0.0015
    symbol: str = 'EURUSD'
    timeframe_minutes: int = 15

    def __post_init__(self):
        if self.look_back < 1 or self.look_forward < 1:
            raise ValueError('Window lengths must be positive')
        if not np.isfinite([self.sl, self.tp]).all() or min(self.sl, self.tp) <= 0:
            raise ValueError('Stop and target distances must be finite and positive')
        if self.symbol != 'EURUSD' or self.timeframe_minutes != 15:
            raise ValueError('This schema supports EURUSD M15 only')


def validate_prices(price):
    price = price.copy()
    price['datetime'] = pd.to_datetime(price['datetime'], errors='raise')
    if price['datetime'].isna().any() or price['datetime'].duplicated().any():
        raise ValueError('Missing or duplicate candle timestamps')
    price = price.sort_values('datetime').reset_index(drop=True)
    price[OHLC] = price[OHLC].apply(pd.to_numeric, errors='raise')
    values = price[OHLC].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('OHLC prices must be finite and positive')
    if ((price.high < price[OHLC].max(axis=1)) |
            (price.low > price[OHLC].min(axis=1))).any():
        raise ValueError('Invalid candle high/low bounds')
    return price


def read_prices(path):
    price = pd.read_csv(path, sep=';', header=None,
                        names=['date', 'time', *OHLC, 'volume'])
    price['datetime'] = pd.to_datetime(price['date'] + ' ' + price['time'],
                                       format='%d/%m/%Y %H:%M:%S', errors='raise')
    return validate_prices(price[['datetime', *OHLC]])


def features(price):
    """EMA uses adjust=False; ATR is Wilder's exponentially smoothed true range."""
    frame = validate_prices(price)
    for period, name in [(100, 'ema_mid'), (200, 'ema_long'), (50, 'ema_short')]:
        frame[name] = frame.close.ewm(span=period, adjust=False, min_periods=period).mean() - frame.close
    previous = frame.close.shift(1)
    true_range = pd.concat([frame.high - frame.low, (frame.high - previous).abs(),
                            (frame.low - previous).abs()], axis=1).max(axis=1)
    frame['atr'] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame[LOG] = np.log(frame[OHLC] / frame[OHLC].shift(1)).to_numpy()
    return frame.dropna(subset=FEATURES).reset_index(drop=True)


def _barrier_labels(entries, highs, lows, sl, tp):
    """Compare first hit positions across independent buy/sell horizons."""
    def first_hit(mask):
        return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1])

    entries = np.asarray(entries)[:, None]
    # Include equality despite one-ULP rounding in decimal price arithmetic.
    buy = first_hit(highs >= np.nextafter(entries + tp, -np.inf)) < first_hit(
        lows <= np.nextafter(entries - sl, np.inf))
    sell = first_hit(lows <= np.nextafter(entries - tp, np.inf)) < first_hit(
        highs >= np.nextafter(entries + sl, -np.inf))
    return np.where(buy & ~sell, 0, np.where(sell & ~buy, 1, 2)).astype('int64')


def label_outcome(entry, future, sl, tp):
    """Stop takes precedence within a candle; ambiguous dual winners become hold."""
    if future.empty:
        return 2
    return int(_barrier_labels([entry], future.high.to_numpy()[None, :],
                              future.low.to_numpy()[None, :], sl, tp)[0])


def label_windows(price, sl, tp, look_forward, look_back, columns_wanted):
    """Compatibility implementation returning (timesteps, outcomes)."""
    Config(look_back=look_back, look_forward=look_forward, sl=sl, tp=tp)
    windows, outcomes = [], []
    for index in range(look_back - 1, len(price) - look_forward):
        windows.append(price[columns_wanted].iloc[index - look_back + 1:index + 1].to_numpy())
        outcomes.append(label_outcome(price.iloc[index]['close'],
                                     price.iloc[index + 1:index + 1 + look_forward], sl, tp))
    return windows, outcomes


def make_dataset(frame, scaler, config):
    if len(frame) < config.look_back + config.look_forward:
        raise ValueError('Partition is too short after indicator warm-up')
    scaled = scaler.transform(frame[FEATURES]).astype('float32')
    count = len(frame) - config.look_back - config.look_forward + 1
    # sliding_window_view avoids allocating all windows before splitting inputs.
    windows = np.lib.stride_tricks.sliding_window_view(scaled, config.look_back, axis=0)
    windows = windows[:count].transpose(0, 2, 1)
    entries = frame.close.to_numpy()[config.look_back - 1:len(frame) - config.look_forward]
    highs = np.lib.stride_tricks.sliding_window_view(
        frame.high.to_numpy()[config.look_back:], config.look_forward)
    lows = np.lib.stride_tricks.sliding_window_view(
        frame.low.to_numpy()[config.look_back:], config.look_forward)
    labels = _barrier_labels(entries, highs, lows, config.sl, config.tp)
    return dict(ta=windows[:, :, :4], ohlc=windows[:, :, 4:],
                labels=np.asarray(labels, dtype='int64'),
                timestamps=frame.datetime.iloc[config.look_back - 1:len(frame) - config.look_forward]
                .to_numpy(dtype='datetime64[ns]').astype('int64'))


def save_metadata(directory, config, scaler):
    # JSON avoids executing pickle payloads when loading a scaler.
    metadata = dict(schema_version=SCHEMA_VERSION, config=asdict(config), features=FEATURES,
                    indicator_method='pandas-ewm-adjust-false-v1',
                    scale=scaler.scale_.tolist(), offset=scaler.min_.tolist(),
                    data_min=scaler.data_min_.tolist(), data_max=scaler.data_max_.tolist(),
                    samples_seen=int(scaler.n_samples_seen_))
    Path(directory, 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def load_metadata(directory):
    metadata = json.loads(Path(directory, 'metadata.json').read_text())
    if (metadata.get('schema_version') != SCHEMA_VERSION or metadata.get('features') != FEATURES
            or metadata.get('indicator_method') != 'pandas-ewm-adjust-false-v1'):
        raise ValueError('Incompatible artifacts: regenerate data and retrain')
    config = Config(**metadata['config'])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for attribute, key in [('scale_', 'scale'), ('min_', 'offset'),
                           ('data_min_', 'data_min'), ('data_max_', 'data_max')]:
        value = np.asarray(metadata[key], dtype=float)
        if value.shape != (8,) or not np.isfinite(value).all():
            raise ValueError('Invalid scaler metadata')
        setattr(scaler, attribute, value)
    if (scaler.scale_ <= 0).any():
        raise ValueError('Invalid scaler scale')
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_features_in_ = 8
    scaler.feature_names_in_ = np.asarray(FEATURES, dtype=object)
    scaler.n_samples_seen_ = metadata['samples_seen']
    return config, scaler


def prepare(csv_path, output, config=Config()):
    output = Path(output)
    # Never mix new preprocessing metadata with an old model.
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; use a new artifact directory')
    price = read_prices(csv_path)
    train_end, validation_end = int(len(price) * .72), int(len(price) * .80)
    frames = [features(part) for part in [price.iloc[:train_end],
              price.iloc[train_end:validation_end], price.iloc[validation_end:]]]
    if any(len(frame) < config.look_back + config.look_forward for frame in frames):
        raise ValueError('Each partition needs indicator warm-up, history, and future candles')
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(frames[0][FEATURES])
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, frame in zip(['train', 'validation', 'test'], frames):
        dataset = make_dataset(frame, scaler, config)
        np.savez_compressed(output / f'{name}.npz', **dataset)
        counts[name] = len(dataset['labels'])
    save_metadata(output, config, scaler)
    return counts


def inference_inputs(price, scaler, config):
    frame = features(price)
    if len(frame) < config.look_back:
        raise ValueError('Insufficient completed candles for indicator warm-up and window')
    values = scaler.transform(frame[FEATURES].tail(config.look_back)).astype('float32')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite model inputs')
    return {'TA': values[None, :, :4], 'OHLC': values[None, :, 4:]}


def load_dataset(directory, name, config):
    with np.load(Path(directory, f'{name}.npz'), allow_pickle=False) as data:
        result = {key: data[key] for key in ['ta', 'ohlc', 'labels', 'timestamps']}
    count = len(result['labels'])
    for key in ['ta', 'ohlc']:
        if result[key].shape != (count, config.look_back, 4) or not np.isfinite(result[key]).all():
            raise ValueError(f'Invalid {name} {key} input shape or values')
    if (not count or result['labels'].shape != (count,)
            or not np.isin(result['labels'], [0, 1, 2]).all()
            or result['timestamps'].shape != (count,)
            or not (np.diff(result['timestamps']) > 0).all()):
        raise ValueError(f'Invalid {name} labels or timestamps')
    return result
