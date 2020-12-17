from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

SELECTION_FUNCTIONS = {}


def register_func(func):
    SELECTION_FUNCTIONS[func.__name__] = func


@register_func
def selectfrommodel(x_train, y_train, problem_type):
    if problem_type == 'regression':
        selector = SelectFromModel(RandomForestRegressor(), threshold='median')
    elif problem_type == 'classification':
        selector = SelectFromModel(RandomForestClassifier(),
                                   threshold='median')
    selector.fit(x_train, y_train)
    x_train_selected = selector.transform(x_train)
    mask = selector.get_support()
    return x_train_selected, mask
