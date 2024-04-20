import pandas as pd
from setting import Config

def load_csv(dir):
    raw_train = pd.read_csv(dir+"DA2024_train.csv")
    raw_test = pd.read_csv(dir+"DA2024_test.csv")
    raw_store = pd.read_csv(dir+"DA2024_stores.csv").drop(["Unnamed: 10", "Unnamed: 11"], axis=1)
    state = pd.read_csv(dir+"store_states.csv")
    state_name = pd.read_csv(dir+"state_names.csv")
    weathers = pd.read_csv(dir+"weather.csv")
    submission = pd.read_csv(dir+"submission.csv")
    Config.test_data_len = len(raw_test)
    return raw_train, raw_test, raw_store, state, state_name, weathers, submission
