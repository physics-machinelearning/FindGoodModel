class InputNanError(Exception):
    """入力値がNanのときあげるエラー"""
    pass


class InputParamError(Exception):
    """param_dictがdictionary出なかった時あげるエラー"""
    pass


class InputProblemTypeError(Exception):
    """problem typeがregressionでもclassificationでもない時のエラー"""
    pass


class InputCrossvalTypeError(Exception):
    """crossval_typeがCROSSVAL_FUNCTIONSに当てはまらない時のエラー"""
    pass


class InputSearchTypeError(Exception):
    """crossval_typeがHYPSEARCH_FUNCTIONSに当てはまらない時のエラー"""
    pass


class InputMetricsError(Exception):
    """metricsが METRICS_FUNCTIONSに当てはまらない時のエラー"""
    pass
