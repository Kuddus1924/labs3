import numpy as np
import pandas
import textwrap

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def save(filename, prediction, grid_cv, **kwargs):
    header = textwrap.dedent(f'''\
        Score: {grid_cv.best_score_}
        Estimator: {grid_cv.best_estimator_}
        Params: {grid_cv.best_params_}''')
    np.savetxt(filename, prediction, header=header, **kwargs)

def read_csv(filename, split=True):
    data = pandas.read_csv(filename, header=None, sep=r',\s*', engine='python')
    if not split:
        return data

    X = data.iloc[:, :-1]
    Y = np.ravel(data.iloc[:, -1:])
    return (X, Y)

def replacement(data,fit_data = None):
    if fit_data is None:
        fit_data = data
    imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
    imputer.fit(fit_data)
    return imputer.transform(data)


def encoderData(number_columns, data, fit_data = None):
    if fit_data is None:
        fit_data = data
    encoder = make_column_transformer((OneHotEncoder(), number_columns), remainder='passthrough')
    encoder.fit(fit_data)
    return encoder.transform(data)


def learning_best_classifier(train,test,classifier,params_classifier,encode_columns,n_jobs):
    train_x,train_y = read_csv(train)
    new_x = replacement(train_x)
    encoded_train_x = encoderData(encode_columns, new_x)
    grid_cv = GridSearchCV(
        classifier,
        params_classifier,
        n_jobs=n_jobs)
    grid_cv.fit(encoded_train_x, train_y)
    test_x = read_csv(test, split=False)
    test_x = replacement(test_x, train_x)
    en_test_x = encoderData(encode_columns, test_x,new_x)
    return grid_cv, grid_cv.best_estimator_.predict(en_test_x)