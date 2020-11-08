import itertools
from sklearn.pipeline import Pipeline
import numpy as np

from bayes_opt import BayesianOptimization

from auto_learning.crossval_func import CROSSVAL_FUNCTIONS

HYPSEARCH_FUNCTIONS = {}


def register_func(func):
    HYPSEARCH_FUNCTIONS[func.__name__] = func


# 全探索
@register_func
def brute(config):
    params_list, keys = _generate_all_combinations(config)   
    score_list = []

    for params in params_list:
        param_dict = {}
        for i, key in enumerate(keys):
            param_dict[key] = params[i]

        if isinstance(config.est, Pipeline):
            est = config.est.set_params(**param_dict)
        else:
            est = config.est(**param_dict)

        score, _, _ = CROSSVAL_FUNCTIONS[config.crossval_type](est, config.x_train, config.y_train, config.metrics)
        score_list.append(score)

    if len(params_list) == 0:
        est = config.est()
    else:
        index = score_list.index(max(score_list))
        param_dict = {}
        params = params_list[index]
        for key, param in zip(keys, params):
            param_dict[key] = param

        if isinstance(config.est, Pipeline):
            est = config.est.set_params(**param_dict)
        else:
            est = config.est(**param_dict)

    r2, y_test_list, y_test_predicted_list = CROSSVAL_FUNCTIONS[config.crossval_type](est, config.x_train, config.y_train, config.metrics)
    return r2, y_test_list, y_test_predicted_list, est


# bayes-optimizationでハイパラ探索
@register_func
def bayes(config):
    def cv(**param_dict):
        new_param_dict = {}
        for item in param_dict.items():
            key, value = item
            flag = log_flags[key]
            int_flag = int_flags[key]
            if flag:
                new_param_dict[key] = np.exp(value)
            else:
                new_param_dict[key] = value
            if int_flag:
                new_param_dict[key] = int(new_param_dict[key])

        if isinstance(config.est, Pipeline):
            est = config.est.set_params(**new_param_dict)
        else:
            est = config.est(**new_param_dict)

        score, _, _ = CROSSVAL_FUNCTIONS[config.crossval_type](est, config.x_train, config.y_train, config.metrics)
        return score

    if len(config.params_dict.keys()) == 0:
        est = config.est()
    else:
        param_range = {}
        log_flags = {}
        int_flags = {}
        for key, values in config.params_dict.items():
            if max(values) > min(values) * 100:
                log_flags[key] = True
                param_range[key] = (np.log(min(values)), np.log(max(values)))
            else:
                log_flags[key] = False
                param_range[key] = (min(values), max(values))
            if all([type(i) == int for i in values]):
                int_flags[key] = True
            else:
                int_flags[key] = False

        bo = BayesianOptimization(cv, param_range, verbose=2, random_state=0)
        bo.maximize(init_points=1, n_iter=20, acq='ei')
        optimized_params = {}
        for key in bo.max['params']:
            value = bo.max['params'][key]
            flag = log_flags[key]
            int_flag = int_flags[key]
            if flag:
                optimized_params[key] = np.exp(value)
            else:
                optimized_params[key] = value
            if int_flag:
                optimized_params[key] = int(optimized_params[key])

        if isinstance(config.est, Pipeline):
            est = config.est.set_params(**optimized_params)
        else:
            est = config.est(**optimized_params)

    r2, y_test_list, y_test_predicted_list = CROSSVAL_FUNCTIONS[config.crossval_type](est, config.x_train, config.y_train, config.metrics)
    return r2, y_test_list, y_test_predicted_list, est


def _generate_all_combinations(config):
    keys = list(config.params_dict.keys())
    params_list = list(config.params_dict.values())

    if len(params_list) == 1:
        params_list = [[params_list[0][i]] for i in range(len(params_list[0]))]
    elif len(params_list) != 0:
        params_list = params_list[0]
        for i in range(1, len(config.params_dict.keys())):
            params_list = list(itertools.product(params_list,
                               list(config.params_dict.values())[i]))
            params_list = [list(temp) for temp in params_list]
            if i >= 2 :
                params_list = [temp[0]+[temp[1]] for temp in params_list]

    return params_list, keys
