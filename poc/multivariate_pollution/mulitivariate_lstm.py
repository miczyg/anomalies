from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Activation
from math import sqrt
from numpy import concatenate

from helpers.data_reader import read_dataframe

IMPORTANT_FEATURES = [16, 12, 19, 2, 40, 37, 28, 1]


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_data(filename):
     # dates + features + labels
    data, labels = read_dataframe(filename, nsamples=5000, usecols=cols_to_use, has_labels=True)
    return data, labels


if __name__ == '__main__':
    # load dataset
    cols_to_use = IMPORTANT_FEATURES + [43]
    # specify the number of lag hours
    lag_hours = 8
    predict_hours = 8
    n_features = 8
    train_dataset = read_csv("../../data/fast_train.csv", usecols=cols_to_use, index_col=0, header=None)
    test_dataset = read_csv("../../data/fast_test.csv", usecols=cols_to_use, index_col=0, header=None)

    frames = [train_dataset, test_dataset]
    dataset = concat(frames)
    print(dataset)
    dataset.columns = ['val1', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain', 'broken']

    # rearrange colums, data to be predicted should be on first position after date!
    cols = dataset.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    dataset = dataset[cols]

    #preprocessing
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, lag_hours, predict_hours)
    print("Reframed shape: ", reframed.shape)

    # split into train and test sets
    values = reframed.values
    n_train_hours = int(0.5 * len(values)) # year of training
    train = values[:n_train_hours]

    # read test dataset
    test = values[n_train_hours:, :]
    # split into input and outputs
    n_obs = (lag_hours + predict_hours) * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print("TrainX shape: {}, trainX len: {}, trainY shape: {}".format(train_X.shape, len(train_X), train_y.shape))
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], lag_hours + predict_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], lag_hours + predict_hours, n_features))
    print("TrainX shape: {}, trainY shape: {}, TestX shape: {}, TestY shape: {}".
          format(train_X.shape, train_y.shape, test_X.shape, test_y.shape))

    EPOCHS = 50
    hidden_neurons = 50
    # design network
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=EPOCHS, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_obs))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    inv_yhat = inv_yhat.astype(int)
    pyplot.plot(inv_yhat, 'o')
    pyplot.plot(inv_y)
    pyplot.show()
