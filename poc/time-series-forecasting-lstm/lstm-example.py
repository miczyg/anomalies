from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import progressbar
from helpers.data_reader import read_dataframe

from bokeh.plotting import figure, output_file, save, show, ColumnDataSource
from bokeh.palettes import Spectral3 as color_palette
from bokeh.models import HoverTool


# sieć dla każdego
# od awarii iść w dwóch kierunkach -> do sprawdzenia co się dzieje dla każdego parametru
#

# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


def real_parser(x):
    return datetime.strptime(x, ' %Y-%m-%d  %H:%M:%S ')


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    bar = progressbar.ProgressBar()
    for i in bar(range(nb_epoch)):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# load dataset
num_rows = 500000
label_col = 42
rows_to_skip = 200000
series, labels = read_dataframe("../../data/training_data.csv", num_rows, usecols=[0, 1, label_col],
                                has_labels=True, **{'skiprows': [i for i in range(200000)]})
print(len(series))
# transform data to be stationary
raw_values = series.value0.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
test_prct = 0.25
test_num = int(test_prct * num_rows)
# split data into train and test-sets
train, test = supervised_values[0:-test_num], supervised_values[-test_num:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 1, 4)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)

# report performance
rmse = sqrt(mean_squared_error(raw_values[-test_num:], predictions))
print('Test RMSE: %.3f' % rmse)

numlines = 2
mypalette = color_palette[0:numlines]

source = ColumnDataSource(data=dict(
    x=series.datetime.values,
    y=raw_values,
    labels=labels,
))

p = figure(x_axis_type="datetime", title="Data predictions on training values",
           width=1080, height=720)

p.line(series.datetime, raw_values, legend="Ground truth", color="#006400")
p.line(series.datetime[-test_num:], predictions, legend="Predictions", color="#FF4500")

lp = p.line('x', 'y', source=source, line_alpha=0.0, line_color="red",
            line_width = 10, legend="Anomaly label")

hover = HoverTool(tooltips=[
        ( 'date',  '@x'),
        ( 'label', '@labels')],
    formatters={
        'date': 'datetime'},
    renderers=[lp],
    mode='vline')

p.add_tools(hover)

output_file("Predictions on real data, {} samples.html".format(num_rows))

# # save the plot
save(p)
show(p)
