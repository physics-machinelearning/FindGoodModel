import numpy as np

from auto_learning.config import Config
from auto_learning.hyp_param_func import HYPSEARCH_FUNCTIONS
from auto_learning.crossval_func import METRICS_FUNCTIONS


class HypParamSearch:
    def __init__(self, x_train, y_train, x_test, y_test, est, problem_type, feature_selection,
                 params_dict, crossval_type, search_type, metrics):
        self.config = Config()
        self.config.est = est
        self.config.problem_type = problem_type
        self.config.feature_selection = feature_selection
        self.config.params_dict = params_dict
        self.config.crossval_type = crossval_type
        self.config.search_type = search_type
        self.config.metrics = metrics
        self.config.y_train = y_train.astype('float')
        self.config.y_test = y_test.astype('float')
        self.config.x_train = x_train.astype('float')
        self.config.x_test = x_test.astype('float')

    def hyp_param_search(self):
        val_score, y_test_list, y_test_predicted_list, est = HYPSEARCH_FUNCTIONS[self.config.search_type](self.config)
        est.fit(self.config.x_train, self.config.y_train)
        y_test_predicted = est.predict(self.config.x_test)
        test_score = METRICS_FUNCTIONS[self.config.metrics](self.config.y_test, y_test_predicted)
        return self.config.y_test, y_test_predicted, val_score, test_score, est

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    # import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.datasets import load_wine, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    import matplotlib.pyplot as plt
    from models import CLASSIFICATION_MODELS, REGRESSION_MODELS

    data = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)

    feature_selection = 'None'
    crossval_type = 'kfold'
    search_type = 'bayes_Gpyopt'
    metrics = 'r2'
    problem_type = 'regression'

    for func in REGRESSION_MODELS.values():
        est, params = func()
        hyp = HypParamSearch(x_train,
                             y_train,
                             x_test,
                             y_test,
                             est,
                             problem_type,
                             feature_selection,
                             params,
                             crossval_type,
                             search_type,
                             metrics)
        y_test_list, y_test_predicted_list, val_score, test_score, est = hyp.hyp_param_search()

        print(est)

        plt.figure()
        plt.scatter(y_test_list, y_test_predicted_list)
        plt.show()

        print(val_score, test_score)

        y_test_predicted_list = np.array(y_test_predicted_list, int)
        cm = confusion_matrix(y_test_list, y_test_predicted_list)

