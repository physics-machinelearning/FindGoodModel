import numpy as np
from sklearn.preprocessing import StandardScaler

from auto_learning.featureselection import SELECTION_FUNCTIONS


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
        self.param_dict = param_dict
        self.problem_type = problem_type
        self.crossval_type = crossval_type
        self.search_type = search_type
        self.mask = None

    @property
    def x_train(self):
        return self.__x_train

    @x_train.setter
    def x_train(self, value):
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
        self.__x_test = self.sc.transform(value[:, self.mask])
