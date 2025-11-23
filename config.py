from datetime import datetime, timedelta
from pathlib import Path

rpath = Path('results')
timeframe = '1h'
datafile = f'binance_BTC_USDT_{timeframe}_20200101.csv.gz'

# Signal optimize function parameters
slag = 12
min_win = 0.025
max_loss = 0.015

# Reconstruct parameters
lag = 3

# PCA parameters
pca_kwargs = {'n_components': None}

# Random Forest parameters
rf_kwargs = dict(
    n_estimators=500,
    max_depth=None,
    # min_weight_fraction_leaf=0.05,
    # class_weight='balanced_subsample',
    # class_weight=weights,
    # random_state=757593,
    n_jobs=-1,
    verbose=0,
    criterion='gini',
    warm_start=True,
)
