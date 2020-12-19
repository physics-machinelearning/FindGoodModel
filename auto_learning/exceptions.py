class InputNanError(Exception):
    """入力値がNanのときあげるエラー"""
    pass


class InputParamError(Exception):
    """param_dictがdictionary出なかった時あげるエラー"""
    pass


class InputProblemtypeError(Exception):
    """problem typeがregressionでもclassificationでもない時のエラー"""
    pass
