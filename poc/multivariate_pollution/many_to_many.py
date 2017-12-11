import sys
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed
from pandas import read_csv

EPOCHS = 5
IMPORTANT_FEATURES = [16, 12, 19, 2, 40, 37, 28, 1]


def create_model(steps_before, steps_after, feature_count):
    """
        creates, compiles and returns a RNN model
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    DROPOUT = 0.5
    LAYERS = 2

    hidden_neurons = 50

    model = Sequential()
    model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False))
    model.add(RepeatVector(steps_after))
    model.add(LSTM(output_dim=hidden_neurons, return_sequences=True))
    model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    return model


def train_sinus(model, dataX, dataY, epoch_count):
    """
        trains only the sinus model
    """


def test_model():
    cols_to_use = IMPORTANT_FEATURES + [43]

    n_pre = 50
    n_post = 10
    ols_to_use = IMPORTANT_FEATURES + [43]
    dataset = read_csv("../../data/fast_train.csv", usecols=cols_to_use, index_col=0, header=0)
    dataset.columns = ['val1', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain', 'broken']
    dataY = np.array(dataset[dataset.columns[3]].values)
    dataX = np.array(range(len(dataY)))
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(n_pre, n_post, 1)
    history = model.fit(dataX, dataY, batch_size=1, nb_epoch=EPOCHS, validation_split=0.05)

    # now test
    dataset = read_csv("../../data/fast_test.csv", usecols=cols_to_use, index_col=0, header=0)
    dataset.columns = ['val1', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain', 'broken']
    testY = np.array(dataset[dataset.columns[3]].values)
    testX = np.array(range(len(testY)))

    predict = model.predict(dataX)

    fig, ax = plt.subplots()

    ind = range(len(dataX) + len(testX))

    ax.plot(ind[:len(dataX)], predict, 'b-x', label='Network input')
    ax.plot(ind[-len(testX):], dataY, 'r-x', label='Many to many model forecast')
    ax.plot(ind[-len(testX):], testY, 'g-x', label='Ground truth')

    plt.xlabel('t')
    plt.ylabel('sin(t)')
    plt.title('Sinus Many to Many Forecast')
    plt.legend(loc='best')
    plt.show()
    plt.cla()


def main():
    test_model()
    return 1


if __name__ == "__main__":
    main()
