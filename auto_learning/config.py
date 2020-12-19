import numpy as np
from sklearn.preprocessing import StandardScaler

from auto_learning.featureselection import SELECTION_FUNCTIONS
from auto_learning.crossval_func import CROSSVAL_FUNCTIONS
from auto_learning.hyp_param_func import HYPSEARCH_FUNCTIONS
from auto_learning.exceptions import (
    InputNanError, InputParamError, InputProblemTypeError,
    InputCrossvalTypeError, InputSearchTypeError
)


class Config:
    """このクラスにデータ、予測モデル、ハイパーパラメータ、評価指標など
    可変な機械学習のパラメータを全て格納
    """
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None,
                 est=None, feature_selection=None, param_dict=None,
                 problem_type=None, crossval_type=None, search_type=None):
        
        self.__x_train = x_train
        self.y_train = y_train
        self.__x_test = x_test
        self.y_test = y_test
        self.est = est
        self.feature_selection = feature_selection
        self.__param_dict = param_dict
        self.__problem_type = problem_type
        self.__crossval_type = crossval_type
        self.__search_type = search_type
        self.mask = None

    @property
    def x_train(self):
        return self.__x_train

    @x_train.setter
    def x_train(self, value):
        if np.isnan(value).all():
            raise InputNanError

        if self.feature_selection != 'None':
            temp, self.mask = SELECTION_FUNCTIONS[self.feature_selection](
                value, self.y_train, self.problem_type
            )
        else:
            temp = value
            mask = np.ones(value.shape[1])
            self.mask = np.array(mask, bool)

        self.sc = StandardScaler()
        self.__x_train = self.sc.fit_transform(temp)

    @property
    def x_test(self):
        return self.__x_test

    @x_test.setter
    def x_test(self, value):
        if np.isnan(value).all():
            raise InputNanError

        self.__x_test = self.sc.transform(value[:, self.mask])

    @property
    def param_dict(self):
        return self.__param_dict

    @param_dict.setter
    def param_dict(self, value):
        if type(value) != dict:
            raise InputParamError
        self.__param_dict = value

    @property
    def problem_type(self):
        return self.__problem_type

    @problem_type.setter
    def problem_type(self, value):
        if value not in ['regression', 'classification']:
            raise InputProblemTypeError
        self.__problem_type = value

    @property
    def crossval_type(self):
        return self.__crossval_type

    @crossval_type.setter
    def crossval_type(self, value):
        if value not in CROSSVAL_FUNCTIONS.keys():
            raise InputCrossvalTypeError
        self.__crossval_type = value

    @property
    def search_type(self):
        return self.__search_type

    @search_type.setter
    def search_type(self, value):
        if value not in HYPSEARCH_FUNCTIONS.keys():
            raise InputSearchTypeError
        self.__search_type = value
