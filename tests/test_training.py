import numpy as np
import pytest

from model import load_bundle, train
from pipeline import inference_inputs, load_metadata, prepare, read_prices


def test_cpu_train_reload_predict(csv_file, tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    import tensorflow as tf
    list_devices = tf.config.list_physical_devices
    monkeypatch.setattr(tf.config, 'list_physical_devices',
                        lambda kind=None: [] if kind == 'GPU' else list_devices(kind))
    output = tmp_path / 'bundle'
    prepare(csv_file, output)
    # Use small chronological subsets for the one-epoch CPU smoke test.
    for name in ['train', 'validation', 'test']:
        path = output / f'{name}.npz'
        with np.load(path) as source:
            arrays = {key: source[key][:16] for key in source.files}
        np.savez_compressed(path, **arrays)
    metrics = train(output, epochs=1, batch_size=8)
    assert all(np.isfinite(list(metrics.values())))
    model, config, scaler = load_bundle(output)
    prediction = model.predict(inference_inputs(read_prices(csv_file), scaler, config), verbose=0)
    assert prediction.shape == (1, 3)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.sum(), 1, atol=1e-5)
    assert model.get_layer('dropout').rate == .2
    assert config == load_metadata(output)[0]
    with (output / 'metadata.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='do not match'):
        load_bundle(output)
