from auto_learning.config import Config
from auto_learning.hyp_param_func import HYPSEARCH_FUNCTIONS
from auto_learning.crossval_func import METRICS_FUNCTIONS


class HypParamSearch:
    def __init__(self, x_train, y_train, x_test, y_test, est, problem_type,
                 feature_selection, params_dict, crossval_type, search_type,
                 metrics):
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
        val_score, y_test_list, y_test_predicted_list, est =\
            HYPSEARCH_FUNCTIONS[self.config.search_type](self.config)
        est.fit(self.config.x_train, self.config.y_train)
        y_test_predicted = est.predict(self.config.x_test)
        test_score = METRICS_FUNCTIONS[self.config.metrics](
            self.config.y_test, y_test_predicted
        )
        return self.config.y_test, y_test_predicted, val_score, test_score, est
