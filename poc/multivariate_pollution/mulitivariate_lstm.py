from keras.callbacks import Callback
from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Activation
from math import sqrt
from numpy import concatenate
from bokeh.plotting import figure, output_file, save, show, ColumnDataSource
from bokeh.palettes import Spectral11 as color_palette
from bokeh.models import HoverTool
from keras.models import load_model
from helpers.data_reader import read_dataframe
from helpers.data_reader import series_to_supervised

import numpy as np
import time

# IMPORTANT_FEATURES = [16, 12, 19, 2, 40, 37, 28, 1]
IMPORTANT_FEATURES = [15, 11, 18, 1, 39, 36, 27, 0]


class IntervalEvaluation(Callback):
    def __init__(self):
        super(Callback, self).__init__()
        self.end = time.time()

    def on_epoch_end(self, epoch, logs={}):
        end = time.time()
        time_diff = end - self.end
        self.end = end
        print("interval evaluation - epoch: {:d} - time: {:.6f}".format(epoch, time_diff))


if __name__ == '__main__':
    # load dataset
    # cols_to_use = IMPORTANT_FEATURES + [43]
    cols_to_use = IMPORTANT_FEATURES + [42]

    # specify the number of lag hours
    back_window = 8
    predict_hours = 8
    n_features = 8
    train_dataset = read_csv("../../data/training_data.csv", usecols=cols_to_use, index_col=0, header=None)
    test_dataset = read_csv("../../data/test_data.csv", usecols=cols_to_use, index_col=0, header=None)

    ival = IntervalEvaluation()

    # join datasets for easier preprocessing
    frames = [train_dataset, test_dataset]
    dataset = concat(frames)
    # print(dataset)

    # some meaningful names to columns
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
    reframed = series_to_supervised(scaled, back_window, predict_hours)
    print("Reframed shape: ", reframed.shape)

    # split into train and test sets
    values = reframed.values
    n_train_hours = len(train_dataset)
    print(n_train_hours)
    train = values[:n_train_hours]

    # prepare datasets for lstm

    # read test dataset
    test = values[n_train_hours:, :]

    # split into input and outputs
    n_obs = (back_window + predict_hours) * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print("TrainX shape: {}, TrainX len: {}, TrainY shape: {}".format(train_X.shape, len(train_X), train_y.shape))

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], back_window + predict_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], back_window + predict_hours, n_features))
    print("TrainX shape: {}, TrainY shape: {}, TestX shape: {}, TestY shape: {}".
          format(train_X.shape, train_y.shape, test_X.shape, test_y.shape))

    # network parameters
    EPOCHS = 15
    hidden_neurons = 100

    # design network
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # model = load_model('my_model.h5')

    # fit network
    history = model.fit(train_X, train_y, epochs=EPOCHS, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False, callbacks=[ival])

    model.save('model_15e_new.h5')

    # plot training history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # numlines = 2
    # mypalette = color_palette[0:numlines]

    # p = figure(x_axis_type="datetime", title="Data predictions on training values",
    #            width=1080, height=720)

    # print(dataset.reset_index()['Date'])
    # p.line(dataset.values[1], history.history['loss'], legend="Ground truth", color="#006400")
    # p.line(dataset.values[1], history.history['val_loss'], legend="Predictions", color="#FF4500")

    # hover = HoverTool(tooltips=[
    #     ('date', '@x'),
    #     ('label', '@labels')],
    #     formatters={
    #         'date': 'datetime'},
    #     renderers=[p],
    #     mode='vline')
    #
    # p.add_tools(hover)
    # show(p)

    # make a prediction
    yhat = model.predict(test_X)
    print(len(yhat))
    print(yhat.shape)
    print(yhat)

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

    # convert to int
    inv_yhat = inv_yhat.astype(int)
    inv_y = inv_y.astype(int)

    # compute accurracy
    subs = np.subtract(inv_yhat, inv_y)
    diff = np.abs(np.sum(subs))
    print("Overal diffrences: {}".format(diff))

    overall_acc = (len(inv_y) - diff) / len(inv_y)
    print("Overall accuracy: {}".format(overall_acc))

    # cut only ones
    pred_ones = np.count_nonzero(inv_yhat)
    test_ones = np.count_nonzero(inv_y)

    anomaly_acc = pred_ones / test_ones
    print("Anomaly accuracy: {}".format(anomaly_acc))

    # plot only if no more than 1000 test values
    if len(test_y) < 10000:
        pyplot.plot(inv_yhat, 'o')
        pyplot.plot(inv_y)
        pyplot.show()
