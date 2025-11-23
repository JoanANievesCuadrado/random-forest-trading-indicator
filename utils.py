import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from datetime import datetime, timedelta, timezone
from joblib import dump, load
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from upsetplot import UpSet, from_indicators
from numba import njit

from config import *

DATAPATH = Path('../data')
UTC = timezone(timedelta())


def load_ccxt_data(symbol='BTC/USDT', timeframe='5m', limit=1000):
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })

    print("⏳ Fetching data from Binance...")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    ts = df['timestamp'].iloc[0]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    filename = f'binance_{symbol.replace("/", "_")}_{timeframe}_{ts}.csv.gz'
    df.to_csv(DATAPATH / filename)
    return df


def load_ccxt_data_v2(symbol='BTC/USDT', timeframe='5m',
                      start_date=datetime(2024, 1, 1, tzinfo=UTC), end_date=None,
                      max_batches=None, pause=0.5):
    """
    Fetch historical OHLCV data from Binance using ccxt with tqdm progress bar.

    Parameters
    ----------
    symbol : str
        e.g. 'BTC/USDT'
    timeframe : str
        e.g. '5m', '1h', '1d'
    start_date : str
        Start date in 'YYYY-MM-DD'
    end_date : str or None
        End date in 'YYYY-MM-DD'; defaults to current UTC time
    max_batches : int or None
        Optional limit on number of requests
    pause : float
        Sleep between requests (seconds)
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(start_date.isoformat())
    tz = start_date.tzinfo

    if end_date is None:
        end_date = datetime.now(tz)

    all_data = []
    limit = 1000
    batch = 0

    # print(f"⏳ Fetching {symbol} ({timeframe}) candles from {start_date} to {end_date.date()}")

    # Rough estimate of how many batches might be needed
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    est_batches = int(((end_date - start_date).total_seconds() * 1000)
                      / (ms_per_candle * limit)) + 1

    # tqdm progress bar
    with tqdm(total=est_batches, desc='Downloading', unit='batch') as pbar:
        while True:
            batch += 1
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            all_data += ohlcv
            last_ts = ohlcv[-1][0]
            since = last_ts + 1  # start after last timestamp

            # update progress
            pbar.update(1)
            last_dt = datetime.fromtimestamp(last_ts/1000, tz).strftime('%Y-%m-%d %H:%M')
            pbar.set_postfix_str(f"Last candle: {last_dt}")

            # stop conditions
            if datetime.fromtimestamp(last_ts/1000, tz) >= end_date:
                break
            if max_batches and batch >= max_batches:
                print("🛑 Reached max_batches limit.")
                break

            time.sleep(pause)  # respect rate limit

    # Build dataframe
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    print(f"✅ Downloaded {len(df)} candles ({len(df)*int(timeframe[:-1])*5/60:.1f} hours approx.)")

    start_str = start_date.strftime('%Y%m%d')
    filename = f'binance_{symbol.replace("/", "_")}_{timeframe}_{start_str}.csv.gz'
    df.to_csv(DATAPATH / filename, compression='gzip')
    print(f"💾 Saved to {filename}")

    return df


def load_local_data(filename='binance_5m.csv.gz'):
    return pd.read_csv(DATAPATH / filename, index_col='timestamp',
                       parse_dates=True)


def load_data():
    return load_ccxt_data()
    # return load_local_data()


def calculate_rsi(series, period=14):
    # Calculate price changes
    delta = series.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate the exponential moving averages (EMA)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Compute RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def show_indicators(df, close, indicators, signal):
    for ind in indicators:
        print(ind)
        fig, ax = plt.subplots()
        ax.plot(close, color='gray', alpha=0.6, label='Price')
        sc = ax.scatter(df.index, close, s=6, alpha=0.6, c=signal, cmap='coolwarm_r')
        plt.colorbar(sc, ax=ax)
        ax1 = ax.twinx()
        ax1.plot(df[ind], c='g', alpha=0.5)
        ax1.scatter(df.index, df[ind], s=6, alpha=0.6, c=signal, cmap='coolwarm_r')
        plt.title(ind)
        plt.tight_layout()
        plt.show()


plus_feat = ['size', 'body', 'light', 'shadow', 'diff_trend_visual_ichimoku_b']

def get_important_feats(rf, columns, lag=3):
    c = np.array([f'{i}-{j}' for i in columns for j in range(lag)])
    
    importance = rf.named_steps['classifier'].estimator_.feature_importances_
    feats_long_ = c[rf.named_steps['classifier'].support_]
    args = np.argsort(-importance)

    feats_df = pd.DataFrame({'feats': feats_long_[args], 'weights': importance[args], 'cumsum': np.cumsum(importance[args])})

    feats_short = []
    for ind in feats_df['feats'].values:
        feats_short.append(ind.rsplit('-', maxsplit=1)[0])
    feats_short = list(set(feats_short))
    print(feats_short)

    return feats_df, feats_short


def get_important_pc(rf):
    support = rf.named_steps['classifier'].support_
    c = np.array([f'PC{i}' for i in range(1, len(support) + 1)])

    importance = rf.named_steps['classifier'].estimator_.feature_importances_
    pcs = c[rf.named_steps['classifier'].support_]
    args = np.argsort(-importance)

    feats_df = pd.DataFrame({
        'pcs': pcs[args],
        'weights': importance[args],
        'cumsum': np.cumsum(importance[args])
    })

    return feats_df


def get_important_feats_pca(rf, pc, columns, lag, th=1):
    components = rf.named_steps['pca'].components_
    vec = components[pc-1]

    c = np.array([f'{i}-{j}' for i in columns for j in range(lag)])
    args = np.argsort(-np.abs(vec))

    feats_df = pd.DataFrame({'feats': c[args], 'weights': vec[args], 'cumsum': np.sqrt(np.cumsum(np.power(vec[args], 2)))})

    feats_short = []
    small_df = feats_df[feats_df['cumsum'] <= th].copy()
    for ind in small_df['feats'].values:
        feats_short.append(ind.rsplit('-', maxsplit=1)[0])
    feats_short = list(set(feats_short))
    print(feats_short)
    return feats_df, feats_short



def plot_feats_vs_cv(rfecv):

    data = {
        key: value
        for key, value in rfecv.cv_results_.items()
        if key in ["n_features", "mean_test_score", "std_test_score"]
    }
    cv_results = pd.DataFrame(data)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()


def split_data(df, y, th=0.6):
    th = int(th*df.shape[0])
    df_train = df.iloc[:th].copy()
    df_test = df.iloc[th:].copy()
    y_train = y[:th].copy()
    y_test = y[th:].copy()
    return df_train, df_test, y_train, y_test


def split_data_v2(df, y, th1=0.5, th2=0.25):
    th1 = int(th1*df.shape[0])
    th2 = th1 + int(th2*df.shape[0])
    df_train = df.iloc[:th1].copy()
    df_test = df.iloc[th1:th2].copy()
    df_rest = df.iloc[th2:].copy()
    y_train = y[:th1].copy()
    y_test = y[th1:th2].copy()
    y_rest = y[th2:].copy()
    return df_train, df_test, df_rest, y_train, y_test, y_rest


def split_data_v3(df: pd.DataFrame, y: pd.Series, td_train: timedelta, td_test: timedelta):
    dt = timedelta(seconds=1)
    t0 = df.index[0]
    t_train = t0 + td_train
    t_test = t_train + td_test

    df_train = df.loc[:t_train].copy()
    df_test = df.loc[t_train + dt:t_test].copy()
    df_rest = df.loc[t_test + dt:].copy()

    y_train = y[:t_train].copy()
    y_test = y[t_train + dt:t_test].copy()
    y_rest = y[t_test + dt:].copy()

    return df_train, df_test, df_rest, y_train, y_test, y_rest


def get_shift(df, lag):
    dtime = df.index[1] - df.index[0]
    lag = timedelta(minutes=lag)
    shift = -int(lag / dtime) + 1
    return shift


def get_signal(X, lag=12):
    df = X.copy()
    df['diff'] = df['close'].diff()
    dtime = df.index[1] - df.index[0]
    lag_t = lag * dtime
    df['meandiff'] = df['diff'].rolling(lag_t).mean()
    th = df['meandiff'].std()

    shift = 1-lag

    long_signal = (df['meandiff'] > th).astype(int).shift(shift, fill_value=0)
    short_signal = (df['meandiff'] < -th).astype(int).shift(shift, fill_value=0)
    signal = long_signal - short_signal
    return signal


def get_signal_v2(X, x, lag=12):
    signal = get_signal(X, lag)
    long_signal = signal.copy()
    short_signal = signal.copy()

    long_signal[long_signal == -1] = 0
    short_signal[short_signal == 1] = 0

    start_long_date = long_signal[long_signal.diff() == 1].index
    end_long_date = long_signal[long_signal.diff() == -1].index
    start_short_date = short_signal[short_signal.diff() == -1].index
    end_short_date = short_signal[short_signal.diff() == 1].index

    signal2 = pd.DataFrame(0, columns=['signal'], index=x.index)
    for i, j in zip(start_long_date, end_long_date):
        signal2[i:j] = 1
    for i, j in zip(start_short_date, end_short_date):
        signal2[i:j] = -1

    return signal, signal2


def run_lag_optimized(close, high, low, i, lag, a, b):
    c = close[i]
    h2 = (1 + a) * c
    h1 = (1 + b) * c
    l1 = (1 - b) * c
    l2 = (1 - a) * c

    # Determine search range
    end = min(len(close), i + lag + 1)
    highs = high[i+1:end]
    lows = low[i+1:end]

    # Vectorized condition checks
    cond_up = (highs > h2) & (lows > l1)
    cond_down = (lows < l2) & (highs < h1)

    if np.any(cond_up):
        return 1
    elif np.any(cond_down):
        return -1
    return 0


def get_signal_optimized(df, lag, a, b):
    close = df['close'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    n = len(df)

    signal = np.zeros(n, dtype=int)
    for i in range(n - 1):
        signal[i] = run_lag_optimized(close, high, low, i, lag, a, b)

    # df = df.copy()
    # df['signal'] = signal
    signal = pd.Series(signal, index=df.index)
    return signal


@njit
def run_lag_numba(close, high, low, lag, a, b):
    n = len(close)
    signal = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        c = close[i]
        h2 = (1 + a) * c
        h1 = (1 + b) * c
        l1 = (1 - b) * c
        l2 = (1 - a) * c

        end = min(n, i + lag + 1)
        for j in range(i + 1, end):
            if high[j] > h2 and low[j] > l1:
                signal[i] = 1
                break
            if low[j] < l2 and high[j] < h1:
                signal[i] = -1
                break
    return signal


def get_signal_numba(df, lag, a, b):
    sig = run_lag_numba(
        df["close"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        lag, a, b
    )
    # out = df.copy()
    # out["signal"] = sig
    sig = pd.Series(sig, index=df.index)
    return sig


def futute_backtesting(df, y_pred, lag):
    df = df.copy()
    df['signal'] = 0
    df.iloc[lag:]['signal'] = y_pred
    # --- STEP 3: Futures Backtest (Leverage + Fees) ---
    initial_balance = 10_000
    leverage = 5
    fee_rate = 0.0004  # 0.04% per side

    balance = initial_balance
    position = 0  # 1 = long, -1 = short
    entry_price = 0

    df["equity"] = 0.0
    df["pnl"] = 0.0

    for i in range(1, len(df)):
        prev_signal = df["signal"].iloc[i - 1]
        open_price = df["open"].iloc[i]

        # --- Open Long ---
        if prev_signal == 1 and position == 0:
            entry_price = open_price
            position = 1
            balance -= balance * fee_rate

        # --- Close Long ---
        elif prev_signal != 1 and position == 1:
            exit_price = open_price
            pnl_pct = (exit_price - entry_price) / entry_price * leverage
            balance = balance * (1 + pnl_pct)
            balance -= balance * fee_rate
            df.loc[df.index[i], "pnl"] = pnl_pct
            position = 0

        # --- Open Short ---
        elif prev_signal == -1 and position == 0:
            entry_price = open_price
            position = -1
            balance -= balance * fee_rate

        # --- Close Short ---
        elif prev_signal != -1 and position == -1:
            exit_price = open_price
            pnl_pct = (entry_price - exit_price) / entry_price * leverage
            balance = balance * (1 + pnl_pct)
            balance -= balance * fee_rate
            df.loc[df.index[i], "pnl"] = pnl_pct
            position = 0

        df.loc[df.index[i], "equity"] = balance

    # --- Close any open position at the last candle ---
    if position != 0:
        final_price = df["close"].iloc[-1]
        if position == 1:
            pnl_pct = (final_price - entry_price) / entry_price * leverage
        else:
            pnl_pct = (entry_price - final_price) / entry_price * leverage
        balance = balance * (1 + pnl_pct)
        balance -= balance * fee_rate
        position = 0

    final_balance = balance

    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:   ${final_balance:,.2f}")
    print(f"Total Return:    {(final_balance/initial_balance - 1)*100:.2f}%")

    # --- STEP 4: Plot Equity Curve ---
    fix, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df["equity"], color="green", label="Strategy Equity")
    ax1 = ax.twinx()
    ax1.plot(df.index, df["close"]/df["close"].iloc[0]*initial_balance, color="gray", label="Buy & Hold (spot)")
    plt.title(f"Crypto Futures Strategy | Leverage={leverage}x | Fee={fee_rate*100:.2f}%")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Account Value (USD)")
    plt.show()


def test(trained, df, y, close, date, lag, l, train):
    td = timedelta(seconds=1)
    df = df.copy()
    # df = df.loc[:datetime(2025, 3, 5)]
    # y = y.loc[:datetime(2025, 3, 5)]
    a = df[df.index < date].shape[0]
    y_pred = trained.predict(df)

    print("Test accuracy:", accuracy_score(y[a+lag:], y_pred[a:]))
    print("\nTest classification report:")
    test_report = classification_report(y[a+lag:], y_pred[a:], digits=3, zero_division=0)
    print(test_report)

    # Confusion matrix
    # fig, ax = plt.subplots()
    cm = confusion_matrix(y[a+lag:], y_pred[a:], labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['short', 'neutral', 'long'])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test Data)")
    plt.tight_layout()
    if train:
        plt.savefig(rpath / f'cm_{l}.png')
        plt.close()

    # df['returns'] = close.pct_change()
    # fee = 0.0004  # 0.04% per side typical on Binance Futures
    # df['strategy_returns'] = df['position'] * df['returns'] - fee * abs(df['position'].diff().fillna(0))

    # #Performance summary
    # df['equity'] = (1 + df['strategy_returns']).cumprod()

    # total_return = df['equity'].iloc[-1] - 1
    # win_rate = (df['strategy_returns'] > 0).sum() / (df['strategy_returns'] != 0).sum()
    # k = timedelta(days=365) / (df.index[1] - df.index[0]) 
    # annualized = (1 + total_return) ** (k / df.shape[0]) - 1

    # print("\n📊 Strategy Backtest Results:")
    # print(f"Total Return: {total_return * 100:.2f}%")
    # print(f"Annualized Return: {annualized * 100:.2f}%")
    # print(f"Win Rate: {win_rate * 100:.2f}%")

    # # Plot
    # fig, ax = plt.subplots()
    # plt.plot(df.index, df['equity'])
    # plt.xlabel('Date')
    # plt.ylabel('Equity (normalized)')
    # plt.grid(True)
    # plt.tight_layout()
    # if train:
    #     plt.savefig(rpath / f'/equity_{l}.png')
    #     plt.close()
    # # plt.show()


    # Plot data with True and predicted labels
    fig1, [ax1 ,ax2] = plt.subplots(2, 1, figsize=(12.5, 6.5))
    ax1.plot(close, c='gray', alpha=0.6, label='BTC price')
    sc = ax1.scatter(close.index, close, c=y, cmap='coolwarm_r')
    ax1.axvline(date, c='k', ls='--')
    fig1.colorbar(sc, ax=ax1)
    ax1.set_title('Test data - True labels')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rate')
    ax1.legend()
    plt.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(figsize=(10, 5))
    df['position'] = 0
    df.loc[df.index[lag:], 'position'] = y_pred
    ax2.plot(close, c='gray', alpha=0.6, label='BTC price')
    sc = ax2.scatter(close.index, close, c=df['position'], cmap='coolwarm_r')
    ax2.axvline(date, c='k', ls='--')
    fig1.colorbar(sc, ax=ax2)
    ax2.set_title('Test data - Predicted labels')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate')
    ax2.legend()
    plt.tight_layout()
    if train:
        plt.savefig(rpath / f'data_{l}.png', dpi=600)
        plt.close()
    else:
        plt.show()

    # futute_backtesting(_df, _y_pred, lag)

def plot_data(df, y):
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    line, = plt.plot(df['close'], c='gray', alpha=0.6, label='BTC price')
    plt.scatter(df.index, df['close'], c=y.values, cmap='coolwarm_r')
    plt.colorbar()
    plt.title('Test data - True labels')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_importance(pipeline, k):
    importances = pipeline.named_steps['classifier'].estimator_.feature_importances_
    plt.figure(figsize=(6, 3))
    dim = len(importances)
    plt.bar(range(dim), importances)
    plt.xlabel("Feature index")
    plt.ylabel("Importance")
    plt.title("Feature importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(rpath / f'importance_{k}.png')
    plt.close()
    # plt.show()



def plot_training(df: pd.DataFrame, y: pd.Series, close, lag: int, l: int, train: bool=False):
    td = timedelta(seconds=1)
    dt = df.index[1] - df.index[0]
    th = l * dt + td
    df = df.copy()
    date = df.loc[df.index[0]+th:].index[0]
    model = load(rpath / f'{l}_rf_pipeline.joblib')

    y_pred = model.predict(df)
    y_prob = model.predict_proba(df)
    df['position'] = 0
    df['short'] = 0.0
    df['hodl'] = 0.0
    df['long'] = 0.0
    df.loc[df.index[lag:], 'position'] = y_pred
    df.loc[df.index[lag:], 'short'] = y_prob[:,0]
    df.loc[df.index[lag:], 'hodl'] = y_prob[:,1]
    df.loc[df.index[lag:], 'long'] = y_prob[:,2]

    # from_ = datetime(2025, 2, 15)  # max(date-timedelta(days=5), datetime(2025, 2, 15))
    # to = max(date+timedelta(days=3), datetime(2025, 3, 30))
    from_ = date-timedelta(days=10)
    to = date+timedelta(days=10)
    slice_ = slice(from_, to)
    df = df.loc[slice_].copy()
    y = y.loc[slice_].copy()

    # Plot data with True and predicted labels
    fig1, [ax1 ,ax2] = plt.subplots(2, 1, figsize=(12.5, 6.5), sharex=True)
    norm = plt.Normalize(vmin=-1, vmax=1)
    ax1.plot(close.loc[df.index], c='gray', alpha=0.6, label='BTC price')
    sc = ax1.scatter(df.index, close.loc[df.index], c=y, cmap='coolwarm_r', norm=norm)
    ax1.axvline(date, c='k', ls='--')
    fig1.colorbar(sc, ax=ax1)
    ax1.set_title('True labels')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rate')
    ax1.legend(loc='lower left')
    plt.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(figsize=(10, 5))
    ax2.plot(close.loc[df.index], c='gray', alpha=0.6, label='BTC price')
    sc = ax2.scatter(df.index, close.loc[df.index], c=df['position'], cmap='coolwarm_r', norm=norm)
    ax2.axvline(date, c='k', ls='--')
    fig1.colorbar(sc, ax=ax2)
    short = df.loc[date, 'short']
    hodl = df.loc[date, 'hodl']
    long_ = df.loc[date, 'long']
    ax2.set_title(f'Predicted labels\n({date})\n(short: {short:.2%}|hold: {hodl:.2%}|long: {long_:.2%})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate')
    ax2.legend(loc='lower left')
    plt.tight_layout()
    if train:
        plt.savefig(rpath / f'short_data_{l}.png')
        plt.close()
    else:
        plt.show()

    return df.loc[date, ['short', 'hodl', 'long']]