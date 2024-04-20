from preprocess.dataframe_preprocessor import preprocess, replace_value
from preprocess.embedding_builder import prepare_data, build_embedding
from setting import Config
from model import Model_map
from util.metrics import rmspe, convert_preds
import numpy as np

if __name__ == "__main__":
    extended_df, submission = preprocess()
    X_train, X_valid, y_train, y_valid, test_df  = prepare_data(extended_df)

    # learn the embedding of the features
    if Config.has_weight_embedding:
        model, weights = build_embedding(X_train, X_valid, y_train, y_valid)
        X_train = replace_value(X_train, weights, model.embeddings)
        X_valid = replace_value(X_valid, weights, model.embeddings)
        test_df = replace_value(test_df, weights, model.embeddings)

    # Prepare data prepare training with decision tree
    X_train = X_train.drop(['Sales', 'Is_Future', 'SalesLog', 'CustomersLog',
                        'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC', 'Max_Sea_Level_PressurehPa', 'Mean_Sea_Level_PressurehPa',

                  'Min_Sea_Level_PressurehPa'], axis=1)


    X_valid = X_valid.drop(['Sales', 'Is_Future', 'SalesLog', 'CustomersLog',
                            'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC', 'Max_Sea_Level_PressurehPa', 'Mean_Sea_Level_PressurehPa',

                    'Min_Sea_Level_PressurehPa'], axis=1)



    final_test_df = test_df.drop(['Sales', 'Is_Future', 'SalesLog', 'CustomersLog',
                            'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC', 'Max_Sea_Level_PressurehPa', 'Mean_Sea_Level_PressurehPa',

                    'Min_Sea_Level_PressurehPa'], axis=1)
    
    preds_list = []
    for model_name in Config.models:
        model = Model_map[model_name](Config.num_rounds)
        # train_x, train_y, valid_x, valid_y
        model.train(X_train,y_train,  X_valid, y_valid)

        preds = model.predict(final_test_df)
        error = rmspe(convert_preds(preds), submission["Sales"])
        print(f"model <{model_name}> rmspe: {error}")

        preds_list.append(preds)
        new_sub = convert_preds(preds, require_df=True)
        
        new_sub.to_csv(f"{Config.output_dir}{model_name}_sub.csv", index=False)




    all_preds = np.mean(preds_list,axis=0)
    all_error = rmspe(convert_preds(all_preds), submission["Sales"])
    print(f"model <ALL> rmspe: {all_error}")
    new_sub = convert_preds(all_preds, require_df=True)
    new_sub.to_csv(f"{Config.output_dir}all_sub.csv", index=False)





