from __future__ import print_function

import numpy as np

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    boston = load_boston()

    # Create a linear regressor and compute CV score
    lr = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    lr_scores = cross_val_score(lr, boston.data, boston.target, cv=10)
    print('Linear regression CV average score: %.6f' % lr_scores.mean())

    # Create a Ridge regressor and compute CV score
    rg = make_pipeline(StandardScaler(with_mean=False), Ridge(0.05))
    rg_scores = cross_val_score(rg, boston.data, boston.target, cv=10)
    print('Ridge regression CV average score: %.6f' % rg_scores.mean())

    # Create a Lasso regressor and compute CV score
    ls = make_pipeline(StandardScaler(with_mean=False), Lasso(0.01))
    ls_scores = cross_val_score(ls, boston.data, boston.target, cv=10)
    print('Lasso regression CV average score: %.6f' % ls_scores.mean())

    # Create ElasticNet regressor and compute CV score
    en = make_pipeline(StandardScaler(with_mean=False), ElasticNet(alpha=0.001, l1_ratio=0.8))
    en_scores = cross_val_score(en, boston.data, boston.target, cv=10)
    print('ElasticNet regression CV average score: %.6f' % en_scores.mean())

    # Find the optimal alpha value for Ridge regression
    rgcv = make_pipeline(StandardScaler(with_mean=False), RidgeCV(alphas=(1.0, 0.1, 0.01, 0.001, 0.005, 0.0025, 0.001, 0.00025)))
    rgcv.fit(boston.data, boston.target)
    print('Ridge optimal alpha: %.3f' % rgcv[1].alpha_)

    # Find the optimal alpha value for Lasso regression
    lscv = make_pipeline(StandardScaler(with_mean=False), LassoCV(alphas=(1.0, 0.1, 0.01, 0.001, 0.005, 0.0025, 0.001, 0.00025)))
    lscv.fit(boston.data, boston.target)
    print('Lasso optimal alpha: %.3f' % lscv[1].alpha_)

    # Find the optimal alpha and l1_ratio for Elastic Net
    encv = make_pipeline(StandardScaler(with_mean=False), ElasticNetCV(alphas=(0.1, 0.01, 0.005, 0.0025, 0.001), l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8)))
    encv.fit(boston.data, boston.target)
    print('ElasticNet optimal alpha: %.3f and L1 ratio: %.4f' % (encv[1].alpha_, encv[1].l1_ratio_))






