# Forex Trading Bot: reproducible training and dry-run execution

An experimental EUR/USD M15 classifier with two LSTM inputs and optional
MetaTrader 5 API execution. Predictions represent buy, sell, or hold; classification
accuracy alone does not establish profitable trading.

## Setup

Use **Python 3.11** in a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m pip check
```

`requirements.txt` contains the core runtime. The development manifest also installs
pytest. Windows users running the MT5 adapter should additionally install
`requirements-windows.txt` and have the MetaTrader 5 terminal installed and logged in.
GUI/OCR automation is disabled and requires no GUI dependencies.

TensorFlow is pinned to 2.16.2 to support this repository's Intel Mac CPU workflow;
[TensorFlow documents 2.16 as the last macOS x86 release series](https://www.tensorflow.org/install/pip).
CPU training is supported. Compatible TensorFlow GPU environments can use detected
GPUs with memory growth enabled. Native Windows GPU setup is not provided here.

## Prepare and train

```bash
python FeatureCreation.py --csv eurusd-15m.csv --output artifacts
python model.py --artifacts artifacts --epochs 100 --batch-size 64
```

The bundled CSV has **no header** and contains semicolon-separated
`date;time;open;high;low;close;volume` columns, with day-first dates. Preparation
validates positive finite prices, candle bounds, and unique timestamps, then sorts
chronologically. Use a new, empty output directory for each experiment.

The versioned schema is:

- 30 completed M15 candles per sample, ending at the entry candle's close.
- Technical input: EMA100-minus-close, EMA200-minus-close, EMA50-minus-close, ATR14.
- Price input: open, high, low, and close log returns, in that order.
- Model calls use named `TA` and `OHLC` inputs. Both inputs have shape `(samples, 30, 4)`; labels are buy `0`, sell `1`, hold `2`.
- EMA uses pandas `ewm(adjust=False)`; ATR uses true range with Wilder smoothing
  (`alpha=1/14`, `adjust=False`). Indicator initialization is part of schema v1.
- Entry is the last input candle's close. The next 15 candles determine outcomes,
  with stop distance `0.0010` and target distance `0.0015`. Equality hits a barrier;
  a stop wins when both barriers occur in one candle. Dual winning directions or
  neither winning direction produce hold. Spread/slippage are not modeled in labels.

Raw candles are split into 72% training, 8% validation, and 20% test. Each partition
warms up indicators independently and contains its own complete windows and label
horizons. Scaling is fitted only on training features. Validation data controls early
stopping; test data is used for the final classification report.

Preparation writes `train.npz`, `validation.npz`, `test.npz`, and `metadata.json`.
Training writes `model.keras` and a `model-manifest.json` checksum manifest beside them. Metadata records feature order, indicator
method, scaler parameters, symbol, timeframe, and labeling settings. Keep the entire
directory together; loading rejects mixed model/metadata files. **Old `.npy` datasets, `.h5` models, and old scalers are incompatible;
regenerate and retrain.** Existing artifact directories are never silently reused by
preparation. Training may replace `model.keras` within the selected directory.

## MT5 dry-run and explicit execution

On Windows with the terminal connected:

```bash
python MetaTrader.py --artifacts artifacts
```

This requests 2,000 completed candles, excluding the forming candle, applies the
saved preprocessing, and previews a request only when buy or sell probability reaches
0.9. EMA is recursive: its initialization can differ slightly when live and historical
history start at different points; the extended history reduces this effect.

Only `--execute` enables submission. Validate with a demo account before enabling it:

```bash
python MetaTrader.py --artifacts artifacts --execute --volume 0.01 --deviation 20
```

Volume is in lots; deviation is in broker points. The adapter checks account access,
symbol settings, tick freshness, price/volume precision, and protective distances.
It uses ask for buys, bid for sells, and the bundle's stop/target distances. It blocks
existing positions or pending orders for the symbol and only supports fill-or-kill
execution. Broker `order_check` must succeed before submission; submission failures
are never automatically retried. An uncertain result requires inspecting the terminal.
This is a single-run script, not a scheduler; do not run concurrent instances.

Order checks follow the [MT5 Python API](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py)
and [symbol filling-policy documentation](https://www.mql5.com/en/book/automation/symbols/symbols_execution_filling).
Broker integration is tested with mocks; actual Windows/demo-terminal validation
remains required. Existing-position checks cannot make concurrent external trading
atomic, and successful submission does not replace ongoing position monitoring.

`python OrderAutomation.py` exits before accessing the terminal. OCR and fixed
coordinates cannot reliably identify positions or confirm protective levels.

## Tests and repository layout

```bash
python -m pytest -q
```

Tests cover label barriers, timestamp/index handling, dataset/scaler alignment,
shared preprocessing, a short CPU training/reload/prediction cycle, and mocked MT5
success/failure paths. Tests never submit broker orders.

`pipeline.py` owns preprocessing and artifact validation. `FeatureCreation.py`,
`model.py`, and `MetaTrader.py` provide guarded CLI entry points.
`AssistanceFunctions.py` and `main.py` retain matching legacy utility definitions;
`main.py` is not a bot launcher. Their label helper delegates to the shared pipeline.
Legacy indicator utilities still require optional `pandas_ta`, which the supported
pipeline does not use. The old standalone custom-loss and experimental generators
are not part of the supported workflow.

## Contributing

Create a feature branch on your fork and submit a PR to
[Kamuyu-N/DeepLearning-Trading-Bot](https://github.com/Kamuyu-N/DeepLearning-Trading-Bot).
Use focused, imperative commits. Describe the affected pipeline stages, behavior
changes, artifact compatibility, and validation results. Keep credentials, local
agent instructions, environments, generated artifacts, and bytecode out of commits.
