import pandas as pd
import numpy as np

# === Basic Utilities ===
def moving_average(series, window):
    return series.rolling(window=window).mean()

def sign_signal(series, threshold=0.0):
    return series.apply(lambda x: +1 if x > threshold else -1 if x < -threshold else 0)


# === 1. Basic Momentum ===
def basic_momentum(df, price_col, window, threshold):
    ma = moving_average(df[price_col], window)
    mt = df[price_col] - ma
    signal = sign_signal(mt, threshold)
    return signal.shift(1)


# === 2. Time-Weighted Momentum ===
def time_weighted_momentum(df, price_col, window, threshold):
    dp = df[price_col].diff()
    weights = np.array([(window - i) / window for i in range(1, window)])
    def weighted_sum(i):
        if i < window:
            return np.nan
        return np.dot(dp.iloc[i-window+1:i+1][::-1].values, weights)
    mt = pd.Series([weighted_sum(i) for i in range(len(df))], index=df.index)
    signal = sign_signal(mt, threshold)
    return signal.shift(1)


# === 3. MA Crossover Momentum ===
def crossover_momentum(df, price_col, fast, slow, threshold):
    ma_fast = moving_average(df[price_col], fast)
    ma_slow = moving_average(df[price_col], slow)
    mt = ma_fast - ma_slow
    signal = sign_signal(mt, threshold)
    return signal.shift(1)


# === 4. Multi-Frequency Momentum ===
def multi_momentum(df, price_col, pairs, weights, threshold):
    signals = [crossover_momentum(df, price_col, fast, slow, threshold) for (fast, slow) in pairs]
    combined = sum(w * s for w, s in zip(weights, signals))
    return sign_signal(combined, threshold)


# === 5. Carry ===
def carry(df, front_col, back_col, threshold):
    ct = df[front_col] - df[back_col]
    signal = sign_signal(ct, threshold)
    return signal.shift(1)


# === 6. Carry Momentum ===
def carry_momentum(df, front_col, back_col, window, threshold):
    ct = df[front_col] - df[back_col]
    ma = moving_average(ct, window)
    signal = sign_signal(ct - ma, threshold)
    return signal.shift(1)


# === 7. Value (Mean Reversion to MA) ===
def value_signal(df, price_col, window, threshold):
    ma = moving_average(df[price_col], window)
    mt = df[price_col] - ma
    signal = -sign_signal(mt, threshold)
    return signal.shift(1)


# === 8. Carry of Carry ===
def carry_of_carry(df, front_col, back_col, threshold):
    carry = df[front_col] - df[back_col]
    coc = carry.diff()
    signal = sign_signal(coc, threshold)
    return signal.shift(1)


# === 9. Congestion Strategy ===
def congestion_strategy(df, spread_col, roll_day_1, roll_day_2):
    signal = pd.Series(index=df.index, dtype='float64')
    signal.iloc[roll_day_1::10] = -1
    signal.iloc[roll_day_2::10] = +1
    return signal.shift(1)


# === 10. Inventory Strategy ===
def inventory_strategy(df, inventory_col, k1, k2, threshold):
    long_term_ma = moving_average(df[inventory_col], k1)
    short_term_ma = moving_average(df[inventory_col], k2)
    deviation = df[inventory_col] - long_term_ma
    mean_revert_signal = deviation.apply(lambda x: +1 if x > threshold else -1 if x < -threshold else 0)
    growth = short_term_ma.diff()
    momentum_signal = -np.sign(growth)
    combined = mean_revert_signal.where(abs(deviation) > threshold, momentum_signal)
    return combined.shift(1)


# Dispatcher
strategy_functions = {
    "Basic Momentum": basic_momentum,
    "Time-Weighted Momentum": time_weighted_momentum,
    "Crossover Momentum": crossover_momentum,
    "Multi-Frequency Momentum": multi_momentum,
    "Carry": carry,
    "Carry Momentum": carry_momentum,
    "Value": value_signal,
    "Carry of Carry": carry_of_carry,
    "Congestion": congestion_strategy,
    "Inventory": inventory_strategy,
}

def get_signal(name, df, **kwargs):
    return strategy_functions[name](df, **kwargs)
