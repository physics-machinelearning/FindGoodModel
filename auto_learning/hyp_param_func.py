import itertools
from sklearn.pipeline import Pipeline
import numpy as np

import GPy
import GPyOpt

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


@register_func
def bayes_Gpyopt(config):

    def cv(params):
        new_param_dict = {}
        for i, bound in enumerate(bounds):
            key = bound['name']
            param = float(params[:, i])
            flag = log_flags[key]
            int_flag = int_flags[key]
            if flag:
                new_param_dict[key] = np.exp(param)
            else:
                new_param_dict[key] = param
            if int_flag:
                new_param_dict[key] = int(new_param_dict[key])

        if isinstance(config.est, Pipeline):
            est = config.est.set_params(**new_param_dict)
        else:
            est = config.est(**new_param_dict)

        score, _, _ = CROSSVAL_FUNCTIONS[config.crossval_type](est, config.x_train, config.y_train, config.metrics)
        return -float(score)

    if len(config.params_dict.keys()) == 0:
        est = config.est()
    else:
        bounds = []
        log_flags = {}
        int_flags = {}
        for i, (key, values) in enumerate(config.params_dict.items()):
            if max(values) > min(values) * 100:
                log_flags[key] = True
                bounds.append(
                    {'name': key,
                    'type': 'continuous',
                    'domain': (np.log(min(values)), np.log(max(values)))
                    }
                )
            else:
                log_flags[key] = False
                bounds.append(
                    {'name': key,
                    'type': 'continuous',
                    'domain': (min(values), max(values))
                    }
                )
            if all([type(j) == int for j in values]):
                int_flags[key] = True
            else:
                int_flags[key] = False

        bo = GPyOpt.methods.BayesianOptimization(cv, bounds)
        bo.run_optimization(max_iter=20)
        optimized_params = {}
        for i, bound in enumerate(bounds):
            key = bound['name']
            value = bo.x_opt[i]
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
