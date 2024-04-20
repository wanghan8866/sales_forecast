class Config:
    test_data_len = -1
    data_dir = "./data/"
    output_dir = "./output/"

    test_ids = None
    sales_max = -1
    customers_max = -1
    test_size = 200000
    random_seed = 44
    has_weight_embedding = True
    num_rounds = 100
    

    models = [
        "lgb", "xgb"
    ]