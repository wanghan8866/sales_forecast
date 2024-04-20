from model.embeddingModel import EntitiyEmbedding
from model.lgbModel import LGBModel
from model.xgbModel import XGBModel

Model_map = {
    "lgb":LGBModel,
    "xgb":XGBModel
}