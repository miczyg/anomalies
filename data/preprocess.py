from pandas import read_csv


def split_for_fast_training(dataset="training_data.csv", train_start=200000,
                            train_num=5000, test_start=518000, test_num=5000, decimal_round=2):
    train_df = read_csv(dataset, nrows=train_num, skiprows=train_start)
    test_df = read_csv(dataset, nrows=test_num, skiprows=test_start)

    train_df.round(decimal_round).to_csv("fast_train.csv")
    test_df.round(decimal_round).to_csv("fast_test.csv")


if __name__ == '__main__':
    split_for_fast_training()
