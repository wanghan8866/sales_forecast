from preprocess.data_loading import load_csv
from setting import Config
import pandas as pd
import numpy as np
import math


def change_date(data_frame):
  df = data_frame.copy()
  df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
  df['Year'] = df['Date'].dt.year
  df['Year'] = df['Year'] - df['Year'].min()
  df['Month'] = df['Date'].dt.month - 1
  df['Day'] = df['Date'].dt.day - 1
  df['DayOfWeek'] = df['Date'].dt.dayofweek
  df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
  df['Days'] = (df['Date'] - df['Date'].min()).dt.days
  df['Days'] = df['Days'] / df['Days'].max()
  df['DayOfMonth'] = [0 if i<=10 else 1 if i<=20 else 2 for i in df['Day']]
  df['QuadYear'] = [0 if i<=13 else 1 if i<=26 else 2 if i<=39 else 3 for i in df['WeekOfYear']]
  return df

def change_label(data_frame, column):
  df = data_frame.copy()
  unique = df[column].unique()
  k = 0
  for str in unique:
    df.loc[df[column] == str, column] = k
    k += 1
  print(column, df[column].dtype, df[column].isna().sum())
  df[column] = df[column].astype(int)
  return df

def replace_value(data_frame, weights, features, drop=True):
  df = data_frame.copy()
  for feature in features:
    df = df.merge(weights[feature], how = "left", on = [feature])
    if drop:
      df = df.drop([feature], axis=1)

  return df

def pre_store(store, state):
  # fill store will medium value
  df = store.copy()
  df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace = True)
  df.fillna(0, inplace = True)
  df = df.merge(state, how='left', on=['Store'])
  return df

def pre_weather(data_frame, state_name, weather):
  df = data_frame.copy()
  weather = weather.rename(columns={'file': 'StateName'})
  weather = weather.merge(state_name, how='left', on=['StateName'])
  weather['Date'] = pd.to_datetime(weather['Date'])
  df = df.merge(weather, how='left', on=['State', 'Date'])
  df['Events'].fillna('NaN', inplace = True)
  df = change_label(df, 'Events')
  df = change_label(df, 'State')
  df['Max_Gust_SpeedKm_h'].fillna(0, inplace=True)
  df['CloudCover'].fillna(df['CloudCover'].median(), inplace = True)
  df['Max_VisibilityKm'].fillna(df['Max_VisibilityKm'].median(), inplace = True)
  df['Mean_VisibilityKm'].fillna(df['Mean_VisibilityKm'].median(), inplace = True)
  df['Min_VisibilitykM'].fillna(df['Min_VisibilitykM'].median(), inplace = True)
  df['CloudCover'].fillna(df['CloudCover'].median(), inplace = True)
  df = df.drop(['Date', 'StateName'], axis=1)
  return df


def pre_preprocess(train, test, store, adding_lag = False):
    train_df = train.copy()
    test_df = test.copy()

    train_df['Is_Future'] = 0

    test_df['Is_Future'] = test_df.index + 1
    test_df.fillna(1, inplace=True)
    extended_df = pd.concat([train_df, test_df], sort=False)
    # print("1, ", len(extended_df))
    extended_df = change_date(extended_df)
    # print("2, ", len(extended_df))
    # remove open is false data
    extended_df = extended_df[(extended_df['Open']!=0) & (extended_df['Sales']!=0)]
    # print("3, ", len(extended_df))
    extended_df = pd.merge(extended_df, store, how = 'inner', on = 'Store')
    # print("4, ", len(extended_df))
    extended_df['Store'] = extended_df['Store'] - 1
    extended_df['CompetitionOpen'] = 12 * (2015 - extended_df.Year - extended_df.CompetitionOpenSinceYear) + \
        (extended_df.Month - extended_df.CompetitionOpenSinceMonth)
    # print("5, ", len(extended_df))
    extended_df['PromoOpen'] = 12 * (2015 - extended_df.Year - extended_df.Promo2SinceYear) + \
        (extended_df.WeekOfYear - extended_df.Promo2SinceWeek) / 4.0
    # print("6, ", len(extended_df))
    extended_df['SalesLog'] = extended_df['Sales'].map(math.log)
    extended_df['CustomersLog'] = extended_df['Customers'].map(math.log)
    extended_df.loc[extended_df['StateHoliday'] == 0, 'StateHoliday'] = '0'
    # print("7, ", len(extended_df))
    extended_df = change_label(extended_df, 'StateHoliday')
    extended_df = change_label(extended_df, 'StoreType')
    extended_df = change_label(extended_df, 'Assortment')
    extended_df = change_label(extended_df, 'PromoInterval')
    # print("8, ", len(extended_df))
    extended_df = extended_df.drop(['Customers', 'Open'], axis=1)
    # print("9, ", len(extended_df))
    if adding_lag:
      extended_df = add_lag(extended_df)
    return extended_df

def add_lag(df):
  df = df.copy()
  target_map = df["Sales"].to_dict()
  sorted_df = df.sort_values(by=["Store", "Date"])
  sorted_df["lag_1_year_ago"] = sorted_df.groupby("Store")["Sales"].shift(364)
  sorted_df["lag_2_month_ago"] = sorted_df.groupby("Store")["Sales"].shift(60)
  sorted_df["lag_2_month_1_ago"] = sorted_df.groupby("Store")["Sales"].shift(61)
  sorted_df["lag_2_month_2_ago"] = sorted_df.groupby("Store")["Sales"].shift(62)
  sorted_df["lag_2_month_3_ago"] = sorted_df.groupby("Store")["Sales"].shift(63)
  sorted_df["lag_2_month_4_ago"] = sorted_df.groupby("Store")["Sales"].shift(64)
  sorted_df["lag_2_month_5_ago"] = sorted_df.groupby("Store")["Sales"].shift(65)
  sorted_df["lag_2_month_6_ago"] = sorted_df.groupby("Store")["Sales"].shift(66)
  sorted_df["lag_2_month_7_ago"] = sorted_df.groupby("Store")["Sales"].shift(67)
  # sorted_df["lag_1_week_ago"] = sorted_df.groupby("Store")["Sales"].shift(7)
  # sorted_df["lag_1_day_ago"] = sorted_df.groupby("Store")["Sales"].shift(1)
  # sorted_df["lag_2_day_ago"] = sorted_df.groupby("Store")["Sales"].shift(2)
  # sorted_df["lag_3_day_ago"] = sorted_df.groupby("Store")["Sales"].shift(3)
  return sorted_df.sort_values(by=["Date", "Store"])


def preprocess():
    raw_train, raw_test, raw_store, state, state_name, weathers, submission = load_csv(Config.data_dir)
    store = pre_store(raw_store, state)
    extended_df = pre_preprocess(raw_train, raw_test, store, adding_lag=False)
    extended_df = pre_weather(extended_df, state_name, weathers)

    Config.sales_max = extended_df['SalesLog'].max()
    Config.customers_max = extended_df['CustomersLog'].max()
    return extended_df, submission
