import numpy as np
import tensorflow as tf
import pandas as pd


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  if isinstance(data, list):
    n_vars = 1
  else:
    n_vars = data.shape[1]
  if isinstance(data, pd.DataFrame):
    pass
  else:
    data = pd.DataFrame(data)
  cols, names = list(), list()
  print(n_vars)
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    print(i)
    cols.append(data.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(data.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  #cols_to_use = names[:len(names) - (n_out)]
  #agg = agg[cols_to_use]
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg
