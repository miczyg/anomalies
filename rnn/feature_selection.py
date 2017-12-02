from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from helpers import data_reader
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import operator


def rfe_selection(X, Y, num_to_select=3):
    model = LogisticRegression()
    rfe = RFE(model, num_to_select)
    fit = rfe.fit(X, Y)
    print("RFE: ")
    print("Num Features: {}".format(fit.n_features_))
    print("Selected Features: {}".format(fit.support_))
    print("Feature Ranking: {}".format(fit.ranking_))
    print("Best {} features: {}".format(num_to_select, fit.ranking_[:num_to_select]))


def pca_selection(X, num_to_select = 3):
    # feature extraction
    pca = PCA(n_components=num_to_select)
    fit = pca.fit(X)
    # summarize components
    print("PCA: ")
    print("Num Features: {}".format(num_to_select))
    print("Explained Variance: {}".format(fit.explained_variance_ratio_))
    # print(fit.components_)


def feature_importance(X, Y, num_to_select = 3):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print("Feature importance: ")
    print(model.feature_importances_)
    d = dict((key, value) for (key, value) in zip(range(len(X)), model.feature_importances_))
    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    sorted_names = [a[0] for a in sorted_d]
    print("Sorted by importance: {}".format(sorted_names))
    print("{} most import features: {}".format(num_to_select, [k for k in sorted_names[:num_to_select]]))

if __name__ == '__main__':
    data, labels = data_reader.read_dataframe("../data/training_data.csv", has_labels=True,
                                              nsamples=1000, **{'skiprows': 202000})
    values = data[data.columns[1:]].values
    # print(values)
    # print(labels)
    X = values
    Y = labels
    features_num = 5
    rfe_selection(X, Y, features_num)

    pca_selection(X, features_num)

    feature_importance(X, Y, features_num)
