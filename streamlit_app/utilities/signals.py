import pandas as pd
import numpy as np

# === Basic Utilities ===
def moving_average(series, window):
    return series.rolling(window=window).mean()

def sign_signal(series, threshold=0.0):
    return series.apply(lambda x: +1 if x > threshold else -1 if x < -threshold else 0)

# MA
def basic_momentum(df, price_col, window, threshold=0.0):
    """
    Signal:
    +1 if (lagged price - MA) > threshold
    -1 if (lagged price - MA) < -threshold
     0 if in between
    """
    lagged_price = df[price_col].shift(1)
    ma = moving_average(lagged_price, window)
    delta = lagged_price - ma

    signal = (delta > threshold).astype(int) * 2 - 1

    return signal

# Carry
def carry(df, front_col, back_col, threshold=0.0):
    """
    Carry signal: +1 if yesterday's F(front) - F(back) > threshold
                  -1 if < -threshold
                  0 if in between
    Buy in backwardation, sell in contango
    """
    spread = (
        pd.to_numeric(df[front_col], errors="coerce").shift(1)
        - pd.to_numeric(df[back_col], errors="coerce").shift(1)
    )
    signal = sign_signal(spread, threshold)
    return signal 

# Value
def value_signal(df, price_col, window, threshold=0.0):
    """
    Value signal is the inverse of momentum:
    Buy when price < MA, sell when price > MA.
    Uses lagged price and MA to avoid lookahead.
    """
    lagged_price = df[price_col].shift(1)
    ma = moving_average(lagged_price, window)
    delta = lagged_price - ma
    signal = -sign_signal(delta, threshold)  # invert momentum signal
    return signal



# === TO IMPLEMENT ===
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


# === 6. Carry Momentum ===
def carry_momentum(df, front_col, back_col, window, threshold):
    ct = df[front_col] - df[back_col]
    ma = moving_average(ct, window)
    signal = sign_signal(ct - ma, threshold)
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

# === 11. Follow-the-Flow (COT Momentum) ===
def follow_the_flow(df, mm_col, m, n):
    ma_m = df[mm_col].rolling(window=m).mean()
    ma_n = df[mm_col].rolling(window=n).mean()
    mt = ma_m - ma_n
    signal = sign_signal(mt)
    return signal.shift(1)


# === 12. Fade-the-Crowded-Trade (COT Sentiment) ===
def fade_crowded_trade(df, mm_col, epsilon, n):
    rolling_max = df[mm_col].rolling(window=n).max()
    rolling_min = df[mm_col].rolling(window=n).min()
    
    SI = (df[mm_col] - rolling_min) / (rolling_max - rolling_min)

    def fade_signal(si_val):
        if si_val <= epsilon:
            return 1
        elif si_val >= 1 - epsilon:
            return -1
        else:
            return 0

    signal = SI.apply(fade_signal)
    return signal.shift(1)

def apply_strategy_returns(df, signal, price_col="Rolling Futures"):
    price = pd.to_numeric(df[price_col], errors="coerce")
    returns = price.diff().fillna(0)
    pnl = signal.shift(1) * returns  # shift signal to avoid lookahead
    return pnl.cumsum()


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
    "Follow-the-Flow": follow_the_flow,
    "Fade-the-Crowded-Trade": fade_crowded_trade,
}

def get_signal(name, df, **kwargs):
    return strategy_functions[name](df, **kwargs)
