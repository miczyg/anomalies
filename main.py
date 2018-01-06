from keras.models import load_model
from helpers.data_reader import series_to_supervised
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

IMPORTANT_FEATURES = [15, 11, 18, 1, 39, 36, 27, 0]

MODEL_PATH = "data/models/model_5k_100e.h5"
TEST_DATA_PATH = "data/test_data.csv"

# TODO: podzielić i puścić dla sliców :)
def main():
    window_size = 8
    cols_to_use = IMPORTANT_FEATURES + [42]
    model = load_model(MODEL_PATH)
    dataset = read_csv(TEST_DATA_PATH, usecols=cols_to_use, index_col=0, header=None)

    # change order to have anomaly in first place
    dataset = rearrange_data(dataset[:100])
    # print(dataset)

    scaler = MinMaxScaler(feature_range=(0, 1))

    dataset, data_y = preprocess_data(dataset, scaler, window_size, 8)

    predicted = model.predict(dataset[:30])

    print("=============PREDICTION RESULT===============")
    print(predicted)
    print("============================")



def preprocess_data(dataset, scaler, window_size, n_features):
    # preprocessing
    values = dataset.values

    # ensure all data is float
    values = values.astype('float32')

    # normalize features
    scaled = scaler.fit_transform(values)

    reframed = series_to_supervised(scaled, window_size, window_size).values

    n_obs = (window_size + window_size) * n_features
    dataset_x, dataset_y = reframed[:, :n_obs], reframed[:, -n_features]

    reshaped = dataset_x.reshape((dataset_x.shape[0], window_size + window_size, n_features))

    return reshaped, dataset_y




def rearrange_data(dataset):
    dataset.columns = ['val1', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain', 'broken']
    # rearrange colums, data to be predicted should be on first position after date!
    cols = dataset.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    return dataset[cols]


if __name__ == '__main__':
    main()