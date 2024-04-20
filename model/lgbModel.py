import lightgbm as lgb
import numpy as np

class LGBModel:
    def __init__(self, num_round):
        self.params = {"objective" : "regression",
          "boosting" : "gbdt",
          "metric" : "rmse",
          "num_iterations" : 7500, #7500
          "top_k" : 30,
          "max_depth" : 8,
          "num_leaves" : 800,
          "min_data_in_leaf" : 20,
          "learning_rate" : 0.02,
          "bagging_fraction" : 0.7,
          "bagging_seed" : 3,
          "bagging_freq" : 5,
          "feature_fraction" : 0.5,
          "num_threads" : 4,
          "verbose":1
         }
        self.dataset_params = {"max_bin" : 200,
                        "min_data_in_bin" : 3
                        }
        self.num_round = num_round
        self.bst = None

    def train(self, train_x, train_y, valid_x, valid_y):
        lgb_train_data = lgb.Dataset(train_x, label=train_y['SalesLog'], params=self.dataset_params)
        lgb_val_data = lgb.Dataset(valid_x, label=valid_y['SalesLog'], params=self.dataset_params)

        self.bst = lgb.train(self.params, lgb_train_data, self.num_round, valid_sets=[
                lgb_train_data, lgb_val_data])
    
    def predict(self, test_data):
        lgb_pred = self.bst.predict(test_data)
        lgb_pred = np.exp(lgb_pred)

        return lgb_pred
        