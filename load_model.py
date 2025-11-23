import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_ta
import ta

from datetime import datetime
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import *
from utils import *


class CustomPCA2(PCA):
    def fit(self, X, y=None):
        data = X.copy()
        data = data.loc[y != 0]
        return self.fit(data, y)


class Restructure(BaseEstimator, TransformerMixin):
    def __init__(self, lag=None):
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.lag:
            return data

        X = X.copy()
        n = self.lag
        data = np.array([X[i - n:i].T.flatten() for i in range(n, len(X))])
        return data

#v6.1
feats_7000_pc2_095 = [
    'momentum_pvo',
    'momentum_pvo_signal',
    'volatility_bbh',
    'volatility_bbl',
    'volatility_bbw',
    'volatility_dch',
    'volatility_dcl',
    'volatility_dcw',
    'volatility_kch',
    'volatility_kcl',
    'volatility_kcw',
    'volatility_atr',  # not vectorice
    'volatility_ui',  # not vectorice
    'trend_adx',  # not vectorice
    'trend_mass_index',
    'light',
    'size',
    'shadow',
]
feats_diff = [
    'close',
    'volatility_bbh',
    'volatility_bbl',
    'volatility_dch',
    'volatility_dcl',
    'volatility_kcl',
    'volatility_kch',
]
def add_ta_features(X):
    X = X.copy()
    columns = ['open', 'high', 'low', 'close', 'volume']
    
    indicator_pvo = ta.momentum.PercentageVolumeOscillator(
        volume=X['volume'],
        window_slow=26,  # default 26
        window_fast=12,  # default 12
        window_sign=9,  # default 9
        fillna=True
    )
    X['momentum_pvo'] = indicator_pvo.pvo()
    X['momentum_pvo_signal'] = indicator_pvo.pvo_signal()

    indicator_bb = ta.volatility.BollingerBands(
        close=X['close'],
        window=20,  # default 20
        window_dev=2,  # default 2
        fillna=True
    )
    X['volatility_bbh'] = indicator_bb.bollinger_hband()
    X['volatility_bbl'] = indicator_bb.bollinger_lband()
    X['volatility_bbw'] = indicator_bb.bollinger_wband()

    indicator_dc = ta.volatility.DonchianChannel(
        high=X['high'], 
        low=X['low'],
        close=X['close'],
        window=20,  # default 20
        offset=0,  # default 0
        fillna=True
    )
    X['volatility_dch'] = indicator_dc.donchian_channel_hband()
    X['volatility_dcl'] = indicator_dc.donchian_channel_lband()
    X['volatility_dcw'] = indicator_dc.donchian_channel_wband()

    indicator_kc = ta.volatility.KeltnerChannel(
        close=X['close'],
        high=X['high'],
        low=X['low'],
        window=10,  # default 10
        fillna=True
    )
    X['volatility_kch'] = indicator_kc.keltner_channel_hband()
    X['volatility_kcl'] = indicator_kc.keltner_channel_lband()
    X['volatility_kcw'] = indicator_kc.keltner_channel_wband()

    X['trend_mass_index'] = ta.trend.mass_index(
        high=X['high'],
        low=X['low'],
        window_fast=9,  # default 9
        window_slow=25,  # default 25
        fillna=True
    )

    # not vectoriced
    X['volatility_atr'] = ta.volatility.average_true_range(
        close=X['close'],
        high=X['high'],
        low=X['low'],
        window=10,  # default 10
        fillna=True
    )
    X['volatility_ui'] = ta.volatility.ulcer_index(
        close=X['close'],
        window=14,  #default 14
        fillna=True
    )
    X['trend_adx'] = ta.trend.adx(
        high=X['high'],
        low=X['low'],
        close=X['close'],
        window=14,  # default 14
        fillna=True
    )

    X['size'] = X['high'] - X['low']
    # X['body'] = X['close'] - X['open']
    X['light'] = X['high'] - X[['open', 'close']].max(axis=1)
    X['shadow'] = X[['open', 'close']].min(axis=1) - X['low']


    # X[feats_diff] = StandardScaler().fit_transform(X[feats_diff])
    X[feats_diff] = X[feats_diff].sub(X['close'].values, axis='rows')

    X = X[feats_7000_pc2_095].copy()

    X = X.fillna(0)

    return X


def plot_candle(rf, df, df1):
    n = 200

    # Slice input for prediction
    df2 = df1.iloc[-n - lag:]

    # Predictions
    y_pred = rf.predict(df2)
    y_pred_p = rf.predict_proba(df2)

    # Convert to DataFrame aligned with df2 index
    yp = pd.DataFrame(y_pred, index=df2.index[-n:], columns=['signal'])
    yp_p = pd.DataFrame(y_pred_p, index=df2.index[-n:], columns=['p_short', 'p_hold', 'p_long'])

    print(yp['signal'].value_counts())

    # Use last n rows
    df3 = df.iloc[-n:]

    # Align predictions to df3 by index intersection
    yp = yp.loc[df3.index]
    yp_p = yp_p.loc[df3.index]

    # Long and short markers — safe when empty
    l = df3['close'].where(yp['signal'] == 1)
    s = df3['close'].where(yp['signal'] == -1)

    ap = []

    # Plot long signals (only if any exist)
    if l.notna().any():
        ap.append(
            mpf.make_addplot(
                l, type='scatter', marker='^', markersize=200, color='blue'
            )
        )

    # Plot short signals (only if any exist)
    if s.notna().any():
        ap.append(
            mpf.make_addplot(
                s, type='scatter', marker='v', markersize=200, color='orange'
            )
        )

    # --- Probability panel (3 lines) ---
    ap.append(
        mpf.make_addplot(
            yp_p['p_long'], panel=1, ylabel='Probabilities', color='blue', secondary_y=False
        )
    )
    ap.append(
        mpf.make_addplot(
            yp_p['p_hold'], panel=1, color='gray', secondary_y=False
        )
    )
    ap.append(
        mpf.make_addplot(
            yp_p['p_short'], panel=1, color='orange', secondary_y=False
        )
    )

    # Plot
    mpf.plot(
        df3,
        type='candle',
        addplot=ap,
        style='tradingview',
        panel_ratios=(3, 1),   # candle panel larger than probability panel
    )


def main():
    df = load_local_data(datafile)
    signal = get_signal_optimized(df, slag, min_win, max_loss)
    df1 = add_ta_features(df)
    df1 = df1.iloc[30:].copy()
    signal = signal.iloc[30:].copy()
    df = df.iloc[30:]
    rf = load(rpath / f'rf_pipeline_{timeframe}.joblib')

    plot_candle(rf, df, df1)



if __name__ == '__main__':
    main()