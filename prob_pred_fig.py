import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

from config import *
from main import *
from utils import *


df = load_local_data(datafile)
signal = get_signal_optimized(df, slag, min_win, max_loss)
close = df['close'].copy()
df = add_ta_features(df)
df = df.iloc[30:].copy()
signal = signal.iloc[30:].copy()
close = close[30:].copy()

file = rpath / 'dfa.csv'
if file.exists():
    df_a = pd.read_csv(file, index_col=0, parse_dates=True)
    start = df_a.shape[0] * step_training + start_training
else:
    df_a = pd.DataFrame(columns=['short', 'hodl', 'long'])
    start = start_training

print(df_a.shape[0])

a = []
for i in tqdm(range(start, end_training, step_training)):
    print(i)
    try:
        a.append(plot_training(df, signal, close, lag, i, True))
    except FileNotFoundError:
        break

df_b = pd.DataFrame(a)
df_a = pd.concat((df_a, df_b))

print(df_a.shape[0])

# df_a = pd.DataFrame(a)
df_a.to_csv(file)

# regions = [(datetime(2025, 2, 20, 20), datetime(2025,2, 24, 12), 'gray')]
# regions.append((regions[-1][1], datetime(2025,2, 25, 20), 'red'))
# regions.append((regions[-1][1], datetime(2025,2, 28, 0), 'gray'))
# regions.append((regions[-1][1], datetime(2025, 3, 6, 0), 'blue'))
# regions.append((regions[-1][1], datetime(2025, 3, 6, 12), 'gray'))
# regions.append((regions[-1][1], datetime(2025, 3, 8, 12), 'red'))
# regions.append((regions[-1][1], df_a.index[-1], 'gray'))


fig, ax1 = plt.subplots()

ax1.plot(df_a['short'], '-o', alpha=0.3, label='short', c='r', ms=3)
ax1.plot(df_a['hodl'], '-o', alpha=0.3, label='hold', c='gray', ms=3)
ax1.plot(df_a['long'], '-o', alpha=0.3, label='long', c='b', ms=3)
ax1.set_ylim((0, 1.1))
plt.legend()

ax = ax1.twinx()
ax.plot(close[df_a.index], '--o', label='Price', c='k', ms=3, alpha=0.5)
# ax.set_ylim((65000, 100000))

# for region in regions:
#     ax.axvspan(xmin=region[0], xmax=region[1], color=region[2], alpha=0.2)


plt.tight_layout()
plt.show()
