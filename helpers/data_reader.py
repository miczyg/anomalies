import datetime
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


if __name__ == '__main__':
    df = pandas.read_csv("../data/training_data.csv", nrows=100)
    selected = df[df.columns[0:2]]

    print(selected)

    df[df.columns[0]] = \
        df[df.columns[0]].map(
            lambda x: datetime.datetime.strptime(
                str(x), ' %Y-%m-%d %H:%M:%S ')
        )
    dates = df[df.columns[0]]
    values = df[df.columns[1]]


    # plot
    plt.plot(dates, values)
    # beautify the x-labels
    plt.gcf().autofmt_xdate()

    plt.show()
