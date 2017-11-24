import datetime
import pandas
import numpy as np

# DATASET = "test"
DATASET = "training"
NUM_SAMPLES = 10000

def get_single_column(file, column, nrows):
    df = pandas.read_csv(file, nrows=nrows)
    return {'dates': df[df.columns[0]],
            'values': df[df.columns[column]]}


def plot_matplot(dataframe):
    '''
    Plots data with matplotlib
    :param dataframe:
    :return:
    '''
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA

    minutes = mdates.MinuteLocator("%M")  # every minute
    hours = mdates.HourLocator("%H")  # every month
    hourFmt = mdates.DateFormatter('%H')

    dates = df[df.columns[0]]
    values1 = df[df.columns[1]]
    values2 = df[df.columns[4]]
    values3 = df[df.columns[5]]
    values = [values1, values2]
    # plotting
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(80, 0))
    par2.axis["right"].toggle(all=True)
    new_fixed_axis = par1.get_grid_helper().new_fixed_axis
    par1.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par1,
                                        offset=(30, 0))
    par1.axis["right"].toggle(all=True)
    host.plot(dates, values3)
    par1.plot(values1)
    par2.plot(values2)
    # beautify the x-labels
    host.format_xdata = mdates.DateFormatter(' %Y-%m-%d  %H:%M:%S ')
    host.xaxis.set_major_locator(hours)
    host.xaxis.set_major_formatter(hourFmt)
    host.xaxis.set_minor_locator(minutes)
    #
    host.grid(True)
    plt.draw()
    plt.show()


def get_y_rangename(val):
    y_ranges = ["default", "hundreds", "thousands", "tt"]
    if -20 <= val <= 20: return y_ranges[0]
    if -100 <= val <= 100: return y_ranges[1]
    if -5000 <= val <= 5000: return y_ranges[2]
    else: return y_ranges[3]


def plot_bokeh(df, labels = None):
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.palettes import Spectral6
    from bokeh.models import LinearAxis, Range1d

    # the number of columns is the number of lines that we will make
    numlines = len(df.columns)

    # import color pallet
    mypalette = Spectral6[0:numlines]

    # make a list of our columns
    col = []
    [col.append(i) for i in df.columns[1:]]

    p = figure(x_axis_type="datetime", title="Sensor values {} data".format(DATASET),
               width=1080, height=720, y_range=(0, 15))
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = "units"
    p.extra_y_ranges = {"hundreds": Range1d(start=-0, end=80),
                        "thousands": Range1d(start=0, end=4000),
                        "tt": Range1d(start=0, end=60000)}
    p.add_layout(LinearAxis(y_range_name="hundreds"), 'left')
    p.add_layout(LinearAxis(y_range_name="thousands"), 'left')
    p.add_layout(LinearAxis(y_range_name="tt"), 'left')

    # loop through our columns and colours
    for (columnnames, colore) in zip(col, mypalette):
        p.line(df.datetime, df[columnnames],
               legend=columnnames,
               color=colore,
               y_range_name=get_y_rangename(df[columnnames][1]))

    # p.line(df.datetime, df.value0)
    # # creates an output file
    output_file("{}_data_{}_samples.html".format(DATASET, NUM_SAMPLES))

    # # save the plot
    save(p)
    show(p)


if __name__ == '__main__':
    label_col = 42
    rows_to_read = NUM_SAMPLES
    cols_to_read = 5
    df = pandas.read_csv("../data/{}_data.csv".format(DATASET), nrows=rows_to_read, header=None)
                         # usecols=[i for i in range(cols_to_read + 1)] + [label_col])
    print(len(df.columns))
    df.columns = ["datetime"] + ["value{}".format(i) for i in range(len(df.columns) - 2)] + ["labels"]
    labels = df["labels"]
    df = df[df.columns[:-1]]
    df.datetime = df.datetime.map(
            lambda x: datetime.datetime.strptime(
                str(x), ' %Y-%m-%d  %H:%M:%S ')
        )
    # data
    # plot_matplot(df)
    plot_bokeh(df, labels)


