"""Train and evaluate the shared two-input model; supports CPU-only machines."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline import load_dataset, load_metadata


def build_model(config):
    from tensorflow import keras
    ta = keras.Input(shape=(config.look_back, 4), name='TA')
    ohlc = keras.Input(shape=(config.look_back, 4), name='OHLC')
    technical = keras.layers.LSTM(64, name='lstm_TA')(ta)
    price = keras.layers.LSTM(128, name='lstm_ohlc')(ohlc)
    price = keras.layers.Dropout(.2, name='dropout')(price)
    x = keras.layers.Concatenate()([technical, price])
    for units in [128, 64, 32]:
        x = keras.layers.Dense(units, activation='relu')(x)
    output = keras.layers.Dense(3, activation='softmax')(x)
    model = keras.Model({'TA': ta, 'OHLC': ohlc}, output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.0003),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(directory, epochs=100, batch_size=64, seed=42):
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report
    if epochs < 1 or batch_size < 1:
        raise ValueError('Epochs and batch size must be positive')
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    config, _ = load_metadata(directory)
    datasets = [load_dataset(directory, name, config) for name in ['train', 'validation', 'test']]
    training, validation, test = datasets
    if any(left['timestamps'][-1] >= right['timestamps'][0]
           for left, right in zip(datasets, datasets[1:])):
        raise ValueError('Dataset partitions overlap or are out of chronological order')
    classes = np.unique(training['labels'])
    weights = compute_class_weight('balanced', classes=classes, y=training['labels'])
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    model = build_model(config)
    def inputs(data):
        return {'TA': data['ta'], 'OHLC': data['ohlc']}
    model.fit(inputs(training), training['labels'], epochs=epochs, batch_size=batch_size,
              shuffle=False, class_weight=class_weights,
              validation_data=(inputs(validation), validation['labels']),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                         restore_best_weights=True)], verbose=2)
    metrics = model.evaluate(inputs(test), test['labels'], return_dict=True, verbose=0)
    predictions = model.predict(inputs(test), verbose=0).argmax(axis=1)
    print(classification_report(test['labels'], predictions, labels=[0, 1, 2],
                                target_names=['buy', 'sell', 'hold'], zero_division=0))
    model.save(Path(directory) / 'model.keras')
    manifest = {name: hashlib.sha256(Path(directory, name).read_bytes()).hexdigest()
                for name in ['metadata.json', 'model.keras']}
    Path(directory, 'model-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return metrics


def load_bundle(directory):
    from tensorflow import keras
    config, scaler = load_metadata(directory)
    manifest = json.loads(Path(directory, 'model-manifest.json').read_text())
    for name in ['metadata.json', 'model.keras']:
        if manifest.get(name) != hashlib.sha256(Path(directory, name).read_bytes()).hexdigest():
            raise ValueError('Model bundle files do not match; regenerate and retrain')
    model = keras.models.load_model(Path(directory) / 'model.keras', compile=False)
    if ([tuple(shape[1:]) for shape in model.input_shape] != [(config.look_back, 4)] * 2
            or tuple(model.output_shape[1:]) != (3,)):
        raise ValueError('Model does not match preprocessing schema; retrain')
    return model, config, scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    print(train(args.artifacts, args.epochs, args.batch_size))


if __name__ == '__main__':
    main()
