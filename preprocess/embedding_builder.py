from model import EntitiyEmbedding
from setting import Config
from sklearn.model_selection import train_test_split

def build_model():
    model = EntitiyEmbedding()
    model.add('Store', input_shape=1115, output_shape=10)
    model.add('DayOfWeek', input_shape=7, output_shape=6)
    model.add('Year', input_shape=3, output_shape=2)
    model.add('Day', input_shape=31, output_shape=10)
    model.add('Month', input_shape=12, output_shape=6)
    model.add('DayOfMonth', input_shape=3, output_shape=2)
    model.add('QuadYear', input_shape=4, output_shape=3)
    model.add('State', input_shape=12, output_shape=6)
    model.add('StateHoliday', input_shape=4, output_shape=3)
    model.add('WeekOfYear', input_shape=53, output_shape=10)
    model.add('Assortment', input_shape=3, output_shape=2)
    model.add('StoreType', input_shape=4, output_shape=3)
    model.add('PromoInterval', input_shape=4, output_shape=3)
    model.add('Events', input_shape=22, output_shape=10)
    model.dense('Promo', output_shape=1)

    # new
    # model.dense('CompetitionDistance', output_shape=1)
    # model.dense('CloudCover',  output_shape=1)

    model.concatenate()
    return model

def prepare_data(data_frame, is_train_random = True):
    df = data_frame.copy()
    train_df = df[df['Is_Future']==0]
    if is_train_random:
        train_df = train_df.sample(frac=1, random_state=Config.random_seed)

    test_df = df[df['Is_Future']!=0]
    Config.test_ids = test_df['Is_Future']

    X = train_df.copy()
    y = train_df[['CustomersLog', 'SalesLog']]
    X_train, X_ee, y_train, y_ee = train_test_split(X, y, test_size=Config.test_size, random_state=Config.random_seed)
    return X_train, X_ee, y_train, y_ee, test_df

def build_embedding(X_train, X_ee, y_train, y_ee ):
    model = build_model()
    model.fit(X_ee, y_ee['CustomersLog']/Config.customers_max,
               X_train, y_train['CustomersLog']/Config.customers_max, epochs=12)
    
    
    return  model, model.get_weight()

