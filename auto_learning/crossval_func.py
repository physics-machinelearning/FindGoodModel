import numpy as np
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score

# CROSSVAL_FUNCTIONSに交差検証用の関数格納。現状kfoldのみ
CROSSVAL_FUNCTIONS = {}
# METRICS_FUNCTIONSに評価関数格納。r2 scoreとaccuracy
METRICS_FUNCTIONS = {}


def register_func(func):
    CROSSVAL_FUNCTIONS[func.__name__] = func


def register_metrics(func):
    METRICS_FUNCTIONS[func.__name__] = func


@register_func
def kfold(est, x, y, metrics):
    kf = KFold(n_splits=3, random_state=44)
    y_test_list = []
    y_test_predict_list = []
    test_indexes = []
    for train, test in kf.split(x):
        est_copy = copy.deepcopy(est)
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        est_copy.fit(x_train, y_train)
        y_test = y_test.flatten()
        y_test_predict = est_copy.predict(x_test).flatten()

        y_test_list.extend(y_test.tolist())
        y_test_predict_list.extend(y_test_predict.tolist())

        test_indexes.extend(test)

    test_indexes = np.array(test_indexes).flatten()
    y_test_list = np.array(y_test_list).flatten()[test_indexes]
    y_test_predict_list = np.array(y_test_predict_list).flatten()[test_indexes]
    score = METRICS_FUNCTIONS[metrics](y_test_list, y_test_predict_list)
    return score, y_test_list, y_test_predict_list


@register_metrics
def r2(y_test_list, y_test_predicted_list):
    return r2_score(y_test_list, y_test_predicted_list)


@register_metrics
def accuracy(y_test_list, y_test_predicted_list):
    return accuracy_score(y_test_list, y_test_predicted_list)
