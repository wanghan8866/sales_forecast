import numpy as np
import pandas as pd
from setting import Config
def rmspe(y_pred, y_test):
    return np.sqrt(np.mean(((y_pred - y_test) / y_test)**2))


def convert_preds(pred, require_df = False):
  pred = pd.DataFrame(pred.T, index=Config.test_ids, columns=['Sales'])
  pred["Id"] = pred.index

  df = pd.DataFrame(range(1, Config.test_data_len+1), columns=['Id'])
  # print(df, pred)
  # df
  df = df.merge(pred, how='left', on=['Id'])
  df.fillna(0, inplace=True)
  # return pred
  if require_df:
    return df
  return df["Sales"].values