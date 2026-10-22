"""One completed-bar prediction; dry-run unless --execute is supplied."""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pipeline import inference_inputs


def choose_direction(prediction, threshold=.9):
    prediction = np.asarray(prediction)
    if (prediction.shape != (1, 3) or not np.isfinite(prediction).all()
            or (prediction < 0).any() or (prediction > 1).any()
            or not np.isclose(prediction.sum(), 1, atol=1e-5)):
        raise ValueError('Invalid model probabilities')
    if not .5 < threshold <= 1:
        raise ValueError('Threshold must be greater than 0.5 and at most 1')
    if prediction[0, 0] >= threshold:
        return 'buy'
    if prediction[0, 1] >= threshold:
        return 'sell'
    return None


def build_request(mt5, config, direction, volume, deviation, now):
    if direction not in ('buy', 'sell'):
        raise ValueError('Unknown order direction')
    if not np.isfinite(volume) or volume <= 0 or deviation < 0:
        raise ValueError('Invalid volume or deviation')
    info = mt5.symbol_info(config.symbol)
    if info is None:
        raise RuntimeError('Symbol information unavailable')
    if not info.visible and not mt5.symbol_select(config.symbol, True):
        raise RuntimeError('Cannot select symbol')
    if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
        raise RuntimeError('Symbol does not allow full trading')
    if (not info.volume_min <= volume <= info.volume_max or info.volume_step <= 0
            or not np.isclose(volume / info.volume_step, round(volume / info.volume_step))):
        raise ValueError('Volume violates broker limits or step')
    tick = mt5.symbol_info_tick(config.symbol)
    if (tick is None or not np.isfinite([tick.ask, tick.bid]).all()
            or tick.bid <= 0 or tick.ask < tick.bid or not 0 <= now - tick.time <= 60):
        raise RuntimeError('Invalid or stale tick')
    if info.point <= 0 or info.trade_tick_size <= 0:
        raise RuntimeError('Invalid symbol precision')
    price = tick.ask if direction == 'buy' else tick.bid
    sign = 1 if direction == 'buy' else -1
    def rounded(value):
        return round(round(value / info.trade_tick_size) * info.trade_tick_size, info.digits)
    sl, tp = rounded(price - sign * config.sl), rounded(price + sign * config.tp)
    reference = tick.bid if direction == 'buy' else tick.ask
    minimum = info.trade_stops_level * info.point
    if min(sl, tp) <= 0 or min(sign * (reference - sl), sign * (tp - reference)) <= 0:
        raise ValueError('Spread or rounding invalidates protective levels')
    if min(sign * (reference - sl), sign * (tp - reference)) < minimum:
        raise ValueError('Protective levels violate broker stop distance')
    # SYMBOL_FILLING_FOK is bit 1; ORDER_FILLING_FOK is an enum, not a flag.
    if not info.filling_mode & 1:
        raise RuntimeError('Symbol does not support fill-or-kill; refusing partial execution')
    return dict(action=mt5.TRADE_ACTION_DEAL, symbol=config.symbol, volume=volume,
                type=mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL,
                price=price, sl=sl, tp=tp, deviation=deviation,
                type_time=mt5.ORDER_TIME_GTC, type_filling=mt5.ORDER_FILLING_FOK,
                magic=302115, comment='DL schema v1')


def ensure_no_position(mt5, symbol):
    positions = mt5.positions_get(symbol=symbol)
    orders = mt5.orders_get(symbol=symbol)
    if positions is None or orders is None:
        raise RuntimeError('Cannot verify existing positions/orders')
    if positions or orders:
        raise RuntimeError('Existing position or pending order for symbol; refusing another order')


def run(mt5, model, config, scaler, execute=False, volume=.6, deviation=20, now=None):
    now = time.time() if now is None else now
    try:
        if not mt5.initialize():
            raise RuntimeError(f'MT5 initialization failed: {mt5.last_error()}')
        account = mt5.account_info()
        if account is None:
            raise RuntimeError('MT5 account unavailable')
        if execute and (not account.trade_allowed or not account.trade_expert):
            raise RuntimeError('Account does not allow automated trading')
        # Start position 1 explicitly excludes the forming candle. Extra history
        # reduces EMA initialization differences relative to historical partitions.
        rates = mt5.copy_rates_from_pos(config.symbol, mt5.TIMEFRAME_M15, 1, 2000)
        if rates is None or len(rates) < 200 + config.look_back:
            raise RuntimeError('Insufficient completed candle history')
        price = pd.DataFrame(rates)
        price['datetime'] = pd.to_datetime(price['time'], unit='s')
        age = now - float(price['time'].max())
        if not config.timeframe_minutes * 60 <= age <= config.timeframe_minutes * 120:
            raise RuntimeError('Latest completed candle is stale or incomplete')
        prediction = model.predict(inference_inputs(price, scaler, config), verbose=0)
        direction = choose_direction(prediction)
        if direction is None:
            return dict(status='hold', probabilities=np.asarray(prediction).tolist())
        ensure_no_position(mt5, config.symbol)
        request = build_request(mt5, config, direction, volume, deviation, now)
        if not execute:
            return dict(status='dry-run', request=request)
        check = mt5.order_check(request)
        if check is None or check.retcode != 0:
            raise RuntimeError(f'Order check failed: {check}')
        ensure_no_position(mt5, config.symbol)
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f'Order submission failed or is uncertain; inspect terminal before retrying: {result}')
        return dict(status='submitted', order=result.order, deal=result.deal, request=request)
    finally:
        mt5.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--execute', action='store_true', help='Submit an order; default only previews it')
    parser.add_argument('--volume', type=float, default=.6)
    parser.add_argument('--deviation', type=int, default=20)
    args = parser.parse_args()
    # Optional Windows and TensorFlow imports occur only after CLI parsing.
    import MetaTrader5 as mt5
    from model import load_bundle
    model, config, scaler = load_bundle(args.artifacts)
    print(run(mt5, model, config, scaler, args.execute, args.volume, args.deviation))


if __name__ == '__main__':
    main()
