import pytest

from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

from auto_learning.models import CLASSIFICATION_MODELS, REGRESSION_MODELS
from auto_learning.hyp_param_search import HypParamSearch


def test_integration_regression():
    feature_selection = 'None'
    crossval_type = 'kfold'
    search_type = 'bayes'
    metrics = 'r2'
    problem_type = 'regression'

    data = load_diabetes()
    x_train, x_test, y_train, y_test =\
        train_test_split(data.data, data.target, test_size=0.3)

    for func in REGRESSION_MODELS.values():
        est, params = func()
        hyp = HypParamSearch(
            x_train,
            y_train,
            x_test,
            y_test,
            est=est,
            problem_type=problem_type,
            feature_selection=feature_selection,
            params_dict=params,
            crossval_type=crossval_type,
            search_type=search_type,
            metrics=metrics
        )
        y_test_list, y_test_predicted_list, val_score, test_score, est \
            = hyp.hyp_param_search()
        r2 = r2_score(y_test_list, y_test_predicted_list)
        assert r2 > 0.4
        assert r2 < 1


def test_integration_classification():
    feature_selection = 'None'
    crossval_type = 'kfold'
    search_type = 'bayes'
    metrics = 'accuracy'
    problem_type = 'classification'

    data = load_wine()
    x_train, x_test, y_train, y_test =\
        train_test_split(data.data, data.target, test_size=0.3)

    for func in CLASSIFICATION_MODELS.values():
        est, params = func()
        hyp = HypParamSearch(
            x_train,
            y_train,
            x_test,
            y_test,
            est=est,
            problem_type=problem_type,
            feature_selection=feature_selection,
            params_dict=params,
            crossval_type=crossval_type,
            search_type=search_type,
            metrics=metrics
        )
        y_test_list, y_test_predicted_list, val_score, test_score, est \
            = hyp.hyp_param_search()
        score = accuracy_score(y_test_list, y_test_predicted_list)
        assert score > 0.4
        assert score < 1
