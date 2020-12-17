# AUTO_LEARNING
機械学習において面倒臭いハイパラチューニングを担うライブラリです。予測モデル(Ridge, SVRなど)と、ハイパラ候補(Ridgeであれば`alpha=[0.0001, 0.001 0.1, 1, 10, 100, 1000]`など)を指定すると、ハイパラチューニングを行い、予測結果を返します。ハイパラ探索には、全探索とベイズ最適化の2つの方法があり、指定できます。

# DEMO
`demo.ipynb`でデモを行なっているのでそちらも参照してください。
logistic回帰のハイパラチューニング

```
# Logisitic回帰をimport
from sklearn.linear_model import LogisticRegression

# Logistic回帰のハイパラ候補を辞書形式で格納
params = {'C': [10**i for i in range(0, 5)]}

# ハイパラ探索時の条件を指定
# 変数選択をしない
feature_selection = 'None'
# クロスバリデーションはkfoldで行う
crossval_type = 'kfold'
# ハイパラ探索はベイズ最適化で行う
search_type = 'bayes_Gpyopt'
# メトリクスはaccuracy
metrics = 'accuracy'
# 分類問題
problem_type = 'classification'

hyp = HypParamSearch(
    x_train,
    y_train,
    x_test,
    y_test,
    est=LogisticRegression,
    problem_type=problem_type,
    feature_selection=feature_selection,
    params_dict=params,
    crossval_type=crossval_type,
    search_type=search_type,
    metrics=metrics
)
y_test_list, y_test_predicted_list, val_score, test_score, est = hyp.hyp_param_search()
```

# Requirement
- numpy
- scikit-learn
- scipy
- GPy
- gpyopt

# Installation
- git clone https://github.com/physics-machinelearning/FindGoodModel2.git
- cd FindGoodModel2
- pip install .
