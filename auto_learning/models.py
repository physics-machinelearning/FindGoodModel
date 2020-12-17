from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

REGRESSION_MODELS = {}
CLASSIFICATION_MODELS = {}


def register_regr(func):
    REGRESSION_MODELS[func.__name__] = func


def register_cla(func):
    CLASSIFICATION_MODELS[func.__name__] = func


@register_regr
def ridge():
    params = {'alpha': [10**i for i in range(-4, 5)]}
    return Ridge, params


@register_regr
def lasso():
    params = {'alpha': [10**i for i in range(-4, 5)]}
    return Lasso, params


@register_regr
def svr():
    params = {'C': [2**i for i in range(-5, 11)],
              'epsilon': [2**i for i in range(-10, 1)],
              'gamma': [2**i for i in range(-20, 11)]}
    return SVR, params


@register_regr
def rfr():
    params = {'n_estimators': [30, 40, 50], 'max_depth': [3, 4, 5, 6]}
    return RandomForestRegressor, params


@register_regr
def pca_ridge():
    params = {
        'pca__n_components': [3, 5, 7, 10],
        'est__alpha': [10**i for i in range(-4, 5)]
    }
    steps = [
        ('pca', PCA()),
        ('est', Ridge())
    ]
    pipeline = Pipeline(steps=steps)
    return pipeline, params


@register_regr
def pca_lasso():
    params = {
        'pca__n_components': [3, 5, 7, 10],
        'est__alpha': [10**i for i in range(-4, 5)]
        }
    steps = [
        ('pca', PCA()),
        ('est', Lasso())
    ]
    pipeline = Pipeline(steps=steps)
    return pipeline, params


@register_cla
def logistic():
    params = {'C': [10**i for i in range(0, 5)]}
    return LogisticRegression, params


@register_cla
def tree():
    params = {}
    return DecisionTreeClassifier, params


@register_cla
def rfc():
    params = {'n_estimators': [100, 200], 'max_depth': [4, 5, 6]}
    return RandomForestClassifier, params


@register_cla
def svc():
    params = {'C': [2**i for i in range(-5, 10)],
              'gamma': [2**i for i in range(-10, 5)]}
    return SVC, params


@register_cla
def knc():
    params = {'n_neighbors': [3, 5, 7, 10, 15]}
    return KNeighborsClassifier, params


@register_cla
def gbc():
    params = {'learning_rate': [0.01, 0.02, 0.05, 0.1],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9, 17],
              'max_features': [0.1, 0.3, 1.0]}
    return GradientBoostingClassifier, params
