import xgboost as xgb
import numpy as np

class XGBModel:
    def __init__(self, num_round):

        self.num_round = num_round
        self.reg = xgb.XGBRegressor(base_score=0.5,
                      booster='gbtree',
                      n_estimators=1300,
                      early_stopping_rounds=150,
                      objective='reg:linear',
                      max_depth=10,
                      subsample = 0.7,
                      colsample_bytree= 0.7,
                      learning_rate=0.05,
                       nthread = 4,
                      enable_categorical = True
                        )

    def train(self, train_x, train_y, valid_x, valid_y):
        self.reg.fit(
        train_x, train_y['SalesLog'],
        eval_set=[(train_x, train_y['SalesLog']), (valid_x, valid_y['SalesLog'])],
        verbose=1)
    
    def predict(self, test_data):
        xg_pred = self.reg.predict(test_data)
        xg_pred = np.exp(xg_pred)
        
        return xg_pred
    