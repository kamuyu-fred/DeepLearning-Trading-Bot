from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from MetaTrader import build_request, choose_direction, run
from pipeline import Config, FEATURES, features


@pytest.fixture
def broker(prices):
    now = prices.datetime.iloc[-1].timestamp() + 901
    mt5 = Mock()
    for name, value in dict(SYMBOL_TRADE_MODE_FULL=4, TRADE_ACTION_DEAL=1,
                            ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, ORDER_TIME_GTC=0,
                            ORDER_FILLING_FOK=0, TRADE_RETCODE_DONE=10009, TIMEFRAME_M15=15).items():
        setattr(mt5, name, value)
    mt5.initialize.return_value = True
    mt5.account_info.return_value = NS(trade_allowed=True, trade_expert=True)
    mt5.symbol_info.return_value = NS(visible=True, trade_mode=4, volume_min=.01, volume_max=10,
                                     volume_step=.01, point=.00001, trade_tick_size=.00001,
                                     digits=5, trade_stops_level=10, filling_mode=1)
    mt5.symbol_info_tick.return_value = NS(ask=1.1001, bid=1.1000, time=now)
    rates = prices.copy()
    rates['time'] = rates.datetime.astype('int64') // 10**9
    mt5.copy_rates_from_pos.return_value = rates.drop(columns='datetime').to_records(index=False)
    mt5.positions_get.return_value = ()
    mt5.orders_get.return_value = ()
    mt5.order_check.return_value = NS(retcode=0)
    mt5.order_send.return_value = NS(retcode=10009, order=1, deal=2)
    return mt5, now


@pytest.fixture
def inputs(prices):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features(prices)[FEATURES])
    model = Mock()
    model.predict.return_value = np.array([[.95, .03, .02]])
    return model, Config(), scaler


def test_default_dry_run(broker, inputs):
    mt5, now = broker
    result = run(mt5, *inputs, now=now)
    assert result['status'] == 'dry-run'
    mt5.copy_rates_from_pos.assert_called_once_with('EURUSD', 15, 1, 2000)
    mt5.order_send.assert_not_called()
    mt5.order_check.assert_not_called()
    mt5.shutdown.assert_called_once()


@pytest.mark.parametrize('direction,price,sl,tp', [('buy', 1.1001, 1.0991, 1.1016),
                                                  ('sell', 1.1000, 1.1010, 1.0985)])
def test_request_prices(broker, direction, price, sl, tp):
    mt5, now = broker
    request = build_request(mt5, Config(), direction, .6, 20, now)
    assert request['price'] == price
    assert request['sl'] == pytest.approx(sl)
    assert request['tp'] == pytest.approx(tp)
    assert request['deviation'] == 20


def test_execute_success(broker, inputs):
    mt5, now = broker
    assert run(mt5, *inputs, execute=True, now=now)['status'] == 'submitted'
    mt5.order_check.assert_called_once()
    mt5.order_send.assert_called_once()
    mt5.shutdown.assert_called_once()


@pytest.mark.parametrize('failure', ['initialize', 'account', 'permission', 'history', 'stale',
                                     'position', 'pending', 'unknown_positions', 'check', 'tick',
                                     'volume', 'stops', 'filling', 'submission'])
def test_failure_blocks_or_reports_submission(broker, inputs, failure):
    mt5, now = broker
    if failure == 'initialize':
        mt5.initialize.return_value = False
    elif failure == 'account':
        mt5.account_info.return_value = None
    elif failure == 'permission':
        mt5.account_info.return_value.trade_expert = False
    elif failure == 'history':
        mt5.copy_rates_from_pos.return_value = None
    elif failure == 'stale':
        now += 3600
    elif failure == 'position':
        mt5.positions_get.return_value = [NS(ticket=1)]
    elif failure == 'pending':
        mt5.orders_get.return_value = [NS(ticket=2)]
    elif failure == 'unknown_positions':
        mt5.positions_get.return_value = None
    elif failure == 'check':
        mt5.order_check.return_value = NS(retcode=1)
    elif failure == 'tick':
        mt5.symbol_info_tick.return_value.time = now - 61
    elif failure == 'volume':
        mt5.symbol_info.return_value.volume_max = .1
    elif failure == 'stops':
        mt5.symbol_info.return_value.trade_stops_level = 1000
    elif failure == 'filling':
        mt5.symbol_info.return_value.filling_mode = 2
    elif failure == 'submission':
        mt5.order_send.return_value = None
    with pytest.raises((RuntimeError, ValueError)):
        run(mt5, *inputs, execute=True, now=now)
    assert mt5.order_send.call_count == (1 if failure == 'submission' else 0)
    mt5.shutdown.assert_called_once()


@pytest.mark.parametrize('prediction', [[[float('nan'), 0, 0]], [[.9, .9, .9]], [[-1, 1, 1]], [1, 0, 0]])
def test_invalid_predictions(prediction):
    with pytest.raises(ValueError):
        choose_direction(prediction)


def test_hold_does_not_send(broker, inputs):
    mt5, now = broker
    inputs[0].predict.return_value = np.array([[.2, .2, .6]])
    assert run(mt5, *inputs, execute=True, now=now)['status'] == 'hold'
    mt5.order_send.assert_not_called()


def test_position_opened_during_check_blocks_send(broker, inputs):
    mt5, now = broker
    mt5.positions_get.side_effect = [(), (NS(ticket=3),)]
    with pytest.raises(RuntimeError, match='Existing position'):
        run(mt5, *inputs, execute=True, now=now)
    mt5.order_check.assert_called_once()
    mt5.order_send.assert_not_called()
    mt5.shutdown.assert_called_once()
