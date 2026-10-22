import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    count = 4000
    close = 1.1 + .002 * np.sin(np.arange(count) / 8) + np.arange(count) * .000001
    return pd.DataFrame(dict(datetime=pd.date_range('2020-01-01', periods=count, freq='15min'),
                             open=close, high=close + .0003, low=close - .0003, close=close))


@pytest.fixture
def csv_file(tmp_path, prices):
    path = tmp_path / 'prices.csv'
    frame = prices.copy()
    frame.insert(0, 'date', frame.datetime.dt.strftime('%d/%m/%Y'))
    frame.insert(1, 'time', frame.datetime.dt.strftime('%H:%M:%S'))
    frame.drop(columns='datetime', inplace=True)
    frame['volume'] = 0
    frame.to_csv(path, sep=';', header=False, index=False)
    return path
