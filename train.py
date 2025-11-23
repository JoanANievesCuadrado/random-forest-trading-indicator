import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_ta
import ta

from joblib import dump
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


def add_pta_features(X):
    X = X.copy()

    # ==== MOMENTUM: PVO ====
    pvo = pandas_ta.pvo(
        volume=X["volume"],
        fast=12,
        slow=26,
        signal=9
    )
    X["momentum_pvo"] = pvo["PVO_12_26_9"]
    X["momentum_pvo_signal"] = pvo["PVOs_12_26_9"]

    # ==== VOLATILITY: Bollinger Bands ====
    bb = pandas_ta.bbands(
        close=X["close"],
        length=20,
        std=2
    )
    X["volatility_bbh"] = bb["BBU_20_2.0_2.0"]
    X["volatility_bbl"] = bb["BBL_20_2.0_2.0"]
    mid = bb['BBM_20_2.0_2.0']
    high = bb["BBU_20_2.0_2.0"]
    low  = bb["BBL_20_2.0_2.0"]
    X["volatility_bbw"] = (high - low) / mid

    # ==== VOLATILITY: Donchian Channel ====
    dc = pandas_ta.donchian(
        high=X["high"],
        low=X["low"],
        close=X["close"],
        lower_length=20,
        upper_length=20
    )
    X["volatility_dch"] = dc["DCU_20_20"]
    X["volatility_dcl"] = dc["DCL_20_20"]
    X["volatility_dcw"] = dc["DCU_20_20"] - dc["DCL_20_20"]

    # ==== VOLATILITY: Keltner Channel ====
    kc = pandas_ta.kc(
        close=X["close"],
        high=X["high"],
        low=X["low"],
        length=10
    )
    X["volatility_kch"] = kc["KCUe_10_2"]
    X["volatility_kcl"] = kc["KCLe_10_2"]
    X["volatility_kcw"] = kc["KCUe_10_2"] - kc["KCLe_10_2"]

    # ==== TREND: Mass Index ====
    X["trend_mass_index"] = pandas_ta.massi(
        high=X["high"],
        low=X["low"],
        fast=9,
        slow=25
    )

    # ==== VOLATILITY: ATR ====
    X["volatility_atr"] = pandas_ta.atr(
        high=X["high"],
        low=X["low"],
        close=X["close"],
        length=10
    )

    # ==== VOLATILITY: Ulcer Index ====
    X["volatility_ui"] = pandas_ta.ui(
        close=X["close"],
        length=14
    )

    # ==== TREND: ADX ====
    X["trend_adx"] = pandas_ta.adx(
        high=X["high"],
        low=X["low"],
        close=X["close"],
        length=14
    )["ADX_14"]

    # ==== Custom candle features ====
    X["size"] = X["high"] - X["low"]
    X["light"] = X["high"] - X[["open", "close"]].max(axis=1)
    X["shadow"] = X[["open", "close"]].min(axis=1) - X["low"]

    # ==== Standard scaling and difference ====
    # X[feats_diff] = StandardScaler().fit_transform(X[feats_diff])
    X[feats_diff] = X[feats_diff].sub(X["close"].values, axis="rows")

    # ==== Select final features ====
    X = X[feats_7000_pc2_095].copy()

    # ==== Handle NA ====
    X = X.fillna(0)

    return X


def get_pipeline_rf(lag, pca_kwargs, rf_kwargs, ):
    pca = CustomPCA2(**pca_kwargs)
    rf = RandomForestClassifier(**rf_kwargs)
    pipeline_rf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('restructure', Restructure(lag=lag)),
        ('pca', pca),
        ('classifier', rf)
    ])

    return pipeline_rf


def full_train(df, df1, signal):
    n_sample = df1.shape[0]
    n_short = signal[signal == -1].shape[0]
    n_neutral = signal[signal == 0].shape[0]
    n_long = signal[signal == 1].shape[0]
    weights = {
        -1: n_sample / (3 * n_short),
        0: 0.9*n_sample / (3 * n_neutral),
        1: n_sample / (3 * n_long),
    }

    weight_long = n_long * weights[1]
    weight_short = n_short * weights[-1]
    weight_hodl = n_neutral * weights[0]
    weight = weight_long + weight_short + weight_hodl
    weight_long_fraction = weight_long / weight / n_long
    weight_short_fraction = weight_short / weight / n_short
    weight_leaf = 3/2 * (weight_long_fraction + weight_short_fraction)

    rf_kwargs['class_weight'] = weights
    rf_kwargs['min_weight_fraction_leaf'] = weight_leaf

    pipeline_rf = get_pipeline_rf(lag=lag, pca_kwargs=pca_kwargs,
                                  rf_kwargs=rf_kwargs)
    pipeline_rf.fit(df1, signal[lag:])
    dump(pipeline_rf, rpath / f'rf_pipeline_{timeframe}.joblib')

    n = 200
    df2 = df1.iloc[-n-lag:]
    y_pred = pipeline_rf.predict(df2)
    yp = pd.DataFrame(y_pred, index=df2.index[-n:])
    print(yp.value_counts())
    df3 = df.iloc[-n:]
    l = df3['close'].where(yp[0]==1)
    s = df3['close'].where(yp[0]==-1)
    ap = [
        mpf.make_addplot(l,
                         type='scatter', marker='^', markersize=200, color='blue'),
        mpf.make_addplot(s,
                         type='scatter', marker='v', markersize=200, color='orange'),
    ]
    mpf.plot(df3, type='candle', addplot=ap, style='tradingview')


def main():
    df = load_local_data(datafile)
    signal = get_signal_optimized(df, slag, min_win, max_loss)
    df1 = add_ta_features(df)
    df1 = df1.iloc[30:].copy()
    signal = signal.iloc[30:].copy()
    df = df.iloc[30:]
    full_train(df, df1, signal)


if __name__ == '__main__':
    main()