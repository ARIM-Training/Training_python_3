"""
grid and hyper parameter auto search program
including the residual plot, feaure importance plot
Linerregression Lasso Decisiontree, Randomforest, Gradientboosting
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.feature_selection import RFE
import scipy as sp

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def grid_liner(X_train, y_train, X_test, y_test, X_name_list):
    """
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list: namelist
    :return:
    """
    Xl = list(range(0, len(X_name_list)))
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # R2スコアー
    print("LinerRegression")
    print("Training Best score : {}".format(reg.score(X_train, y_train)))
    print("Test Best score : {}".format(reg.score(X_test, y_test)))

    # 残差プロット
    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)

    train = plt.scatter(pred_train, (pred_train - y_train), c='b', alpha=0.5)
    test = plt.scatter(pred_test, (pred_test - y_test), c='r', alpha=0.5)
    plt.legend((train, test), ('Training', 'Test'), loc='upper left')
    plt.title('LinerResidual Plots')
    plt.xlabel("Label data")
    plt.ylabel("Predict-Answer")
    plt.show()

    # 使用された特徴量
    print("Number of features used:\n{}".format(np.sum(reg.coef_ != 0)))
    plt.plot(reg.coef_, 's')
    plt.title("Liner features")
    plt.xlabel("Cooefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # RFE
    n_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for ii in n_features:
        rfe = RFE(reg,n_features_to_select=ii)
        rfe.fit(X_train, y_train)
    # visualize the selected features:
        mask_rfe = rfe.get_support()

        plt.matshow(mask_rfe.reshape(1, -1), cmap='gray_r')
        plt.yticks(())
        plt.xticks(Xl, X_name_list, rotation=90)
        plt.xlabel("RFE Liner n={}".format(ii))
        plt.show()

        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)

        print("LinearRegressionn={}".format(ii))
        reg3 = LinearRegression()
        reg3.fit(X_train_rfe, y_train)
        print("Training Best score : {:.3f}".format(reg3.score(X_train_rfe, y_train)))
        print("Test Best score : {:.3f}".format(reg3.score(X_test_rfe, y_test)))

    print()
    return reg

def grid_lasso(X_train, y_train, X_test, y_test, X_name_list):
    '''
     GridSearch lasso
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :param params:
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))

    from sklearn.linear_model import Lasso
    # 正則化パラメータ
    params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    # Crossvaridationを5回、すべてのコアを使う
    grid = GridSearchCV(Lasso(max_iter=100000), param_grid=params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # CVを制御したいとき　クラス分類ではStratifyKFold,回帰ではKFold
    # from sklearn.model_selection import KFold
    # GridSearchCVの引数に　cv=KFold(n_splits=5, shuffle=True, random_state=1)

    # R2スコアー
    print("LassoRegression")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))
    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    # print("Number of features used: {}".format(np.sum(grid.coef_ != 0)))

    # 残差プロット
    pred_train = grid.predict(X_train)
    pred_test = grid.predict(X_test)

    train = plt.scatter(pred_train, (pred_train - y_train), c='b', alpha=0.5)
    test = plt.scatter(pred_test, (pred_test - y_test), c='r', alpha=0.5)
    plt.legend((train, test), ('Training', 'Test'), loc='upper left')
    plt.title('Lasso Residual Plots')
    plt.xlabel("Label data")
    plt.ylabel("Predict-Answer")
    plt.show()

    # 使用された特徴量
    # GridSearchCVはcoef_メソッド持っていないためベストの値でインスタンスを生成する。
    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    print("Number of features used:\n{}".format(np.sum(reg.coef_ != 0)))
    plt.plot(reg.coef_, 's')
    plt.title("Lasso features")
    plt.xlabel("Cooefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # RFE
    n_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for ii in n_features:
        rfe = RFE(reg, n_features_to_select=ii)
        rfe.fit(X_train, y_train)
        # visualize the selected features:
        mask_rfe = rfe.get_support()

        plt.matshow(mask_rfe.reshape(1, -1), cmap='gray_r')
        plt.yticks(())
        plt.xticks(Xl, X_name_list, rotation=90)
        plt.xlabel("RFE Lasso n={}".format(ii))
        plt.show()

        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)

        print("Lasso n={}".format(ii))
        reg3 = grid.best_estimator_
        reg3.fit(X_train_rfe, y_train)
        print("Training Best score : {:.3f}".format(reg3.score(X_train_rfe, y_train)))
        print("Test Best score : {:.3f}".format(reg3.score(X_test_rfe, y_test)))
    print()
    return reg

def grid_decisiontree(X_train, y_train, X_test, y_test, X_name_list):
    '''

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))
    # GridSearch DecisionTreeRegressor
    from sklearn.tree import DecisionTreeRegressor

    # 正則化パラメータ
    params = {'max_depth': [2, 3, 4, 5]}

    # Crossvaridationを5回、すべてのコアを使う
    grid = GridSearchCV(DecisionTreeRegressor(), param_grid=params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # R2スコアー
    print("DecisionTree")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))
    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    print("DT Best estimater: \n{}".format(grid.best_estimator_))

    # 残差プロット
    pred_train = grid.predict(X_train)
    pred_test = grid.predict(X_test)

    train = plt.scatter(pred_train, (pred_train - y_train), c='b', alpha=0.5)
    test = plt.scatter(pred_test, (pred_test - y_test), c='r', alpha=0.5)
    plt.legend((train, test), ('Training', 'Test'), loc='lower left')
    plt.title('DecisionTree Residual Plots')
    plt.xlabel("Label data")
    plt.ylabel("Predict-Answer")
    plt.show()

    # 決定木の特徴量の重要性
    # GridSearchCVはfeature_importances_メソッド持っていないためベストの値でインスタンスを生成する。
    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    print("Feature importances:\n{}".format(reg.feature_importances_))
    plt.plot(reg.feature_importances_, 's')
    plt.title("DecisionTree feature importances")
    plt.xlabel("Label data")
    plt.ylabel("Importance")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # RFE
    n_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for ii in n_features:
        rfe = RFE(reg, n_features_to_select=ii)
        rfe.fit(X_train, y_train)
        # visualize the selected features:
        mask_rfe = rfe.get_support()

        plt.matshow(mask_rfe.reshape(1, -1), cmap='gray_r')
        plt.yticks(())
        plt.xticks(Xl, X_name_list, rotation=90)
        plt.xlabel("RFE DecisionTree n={}".format(ii))
        plt.show()

        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)

        print("DecisionTree n={}".format(ii))
        reg3 = grid.best_estimator_
        reg3.fit(X_train_rfe, y_train)
        print("Training Best score : {:.3f}".format(reg3.score(X_train_rfe, y_train)))
        print("Test Best score : {:.3f}".format(reg3.score(X_test_rfe, y_test)))
    print()
    return reg


def grid_randomforest(X_train, y_train, X_test, y_test, X_name_list):
    '''
     GridSearch RandomForest
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))

    params = {'n_estimators': [5, 10, 50, 100, 200]}
    # estimatorsは大きければ大きい程よいが、より多くの決定木の平均値をとり過剰適合が低減される。訓練時間が多くかかることが問題。
    grid = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # R2スコアー
    print("RandomForest")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))
    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    print("RF Best estimater: \n{}".format(grid.best_estimator_))

    # 残差プロット
    pred_train = grid.predict(X_train)
    pred_test = grid.predict(X_test)

    train = plt.scatter(pred_train, (pred_train - y_train), c='b', alpha=0.5)
    test = plt.scatter(pred_test, (pred_test - y_test), c='r', alpha=0.5)
    plt.legend((train, test), ('Training', 'Test'), loc='lower left')
    plt.title('RandomForest Residual Plots')
    plt.xlabel("Label data")
    plt.ylabel("Predict-Answer")
    plt.show()

    # RandomForestの特徴量の重要性
    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    print("Feature importances:\n{}".format(reg.feature_importances_))
    plt.plot(reg.feature_importances_, 's')
    plt.title("RandomForest feature importances")
    plt.xlabel("Label data")
    plt.ylabel("Importance")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # RFE
    n_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for ii in n_features:
        rfe = RFE(reg, n_features_to_select=ii)
        rfe.fit(X_train, y_train)
        # visualize the selected features:
        mask_rfe = rfe.get_support()

        plt.matshow(mask_rfe.reshape(1, -1), cmap='gray_r')
        plt.yticks(())
        plt.xticks(Xl, X_name_list, rotation=90)
        plt.xlabel("RFE RandomForest n={}".format(ii))
        plt.show()

        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)

        print("RandomForest n={}".format(ii))
        reg3 = grid.best_estimator_
        reg3.fit(X_train_rfe, y_train)
        print("Training Best score : {:.3f}".format(reg3.score(X_train_rfe, y_train)))
        print("Test Best score : {:.3f}".format(reg3.score(X_test_rfe, y_test)))
    print()
    return reg


def grid_gradientboosting(X_train, y_train, X_test, y_test, X_name_list):
    '''
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))
    # GradientBoosting

    params = {'max_depth': [1, 2, 3, 4, 5], 'n_estimators': [5, 10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

    grid = GridSearchCV(GradientBoostingRegressor(), param_grid=params, cv=5, n_jobs=-1)

    grid.fit(X_train, y_train)

    # R2スコアー
    print("GradientBoosting")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))
    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    print("RF Best estimater: \n{}".format(grid.best_estimator_))

    # 残差プロット
    pred_train = grid.predict(X_train)
    pred_test = grid.predict(X_test)
    train = plt.scatter(pred_train, (pred_train - y_train), c='b', alpha=0.5)
    test = plt.scatter(pred_test, (pred_test - y_test), c='r', alpha=0.5)
    plt.legend((train, test), ('Training', 'Test'), loc='upper left')
    plt.title('GradientBoosting Residual Plots')
    plt.xlabel("Label data")
    plt.ylabel("Predict-Answer")
    plt.show()

    # GradientBoostingの特徴量の重要性
    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    print("Feature importances:\n{}".format(reg.feature_importances_))
    plt.plot(reg.feature_importances_, 's')
    plt.title("GradientBoosting feature importances")
    plt.xlabel("Label data")
    plt.ylabel("Importance")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # RFE
    n_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for ii in n_features:
        rfe = RFE(reg, n_features_to_select=ii)
        rfe.fit(X_train, y_train)
        # visualize the selected features:
        mask_rfe = rfe.get_support()

        plt.matshow(mask_rfe.reshape(1, -1), cmap='gray_r')
        plt.yticks(())
        plt.xticks(Xl, X_name_list, rotation=90)
        plt.xlabel("RFE GradientBoosting n={}".format(ii))
        plt.show()

        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)

        print("GradientBoosting n={}".format(ii))
        reg3 = grid.best_estimator_
        reg3.fit(X_train_rfe, y_train)
        print("Training Best score : {:.3f}".format(reg3.score(X_train_rfe, y_train)))
        print("Test Best score : {:.3f}".format(reg3.score(X_test_rfe, y_test)))

    print()
    return reg


def grid_svm(X_train, y_train, X_test, y_test, X_name_list):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param Xl:
    :param X_name_list:
    :return:
    """
    Xl = list(range(0, len(X_name_list)))
    from sklearn.svm import SVR
    params = {'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]}

    grid = GridSearchCV(SVR(), param_grid=params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # R2スコアー
    print("SVM Regression")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))
    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    # print("Number of features used: {}".format(np.sum(grid.coef_ != 0)))

    # 残差プロット
    pred_train = grid.predict(X_train)
    pred_test = grid.predict(X_test)

    train = plt.scatter(pred_train, (pred_train - y_train), c='b', alpha=0.5)
    test = plt.scatter(pred_test, (pred_test - y_test), c='r', alpha=0.5)
    plt.legend((train, test), ('Training', 'Test'), loc='upper left')
    plt.title('SVM Residual Plots')
    plt.xlabel("Label data")
    plt.ylabel("Predict-Answer")
    plt.show()

    return grid


def grid_kneighbors(X_train, y_train, X_test, y_test,):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    from sklearn.neighbors import KNeighborsRegressor
    params = {'n_neighbors': [1, 2, 3, 4, 5]}
    grid = GridSearchCV(KNeighborsRegressor(), param_grid=params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # R2スコアー
    print("KNeighborsRegresioon")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))
    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    # print("Number of features used: {}".format(np.sum(grid.coef_ != 0)))

    # 残差プロット
    pred_train = grid.predict(X_train)
    pred_test = grid.predict(X_test)

    train = plt.scatter(pred_train, (pred_train - y_train), c='b', alpha=0.5)
    test = plt.scatter(pred_test, (pred_test - y_test), c='r', alpha=0.5)
    plt.legend((train, test), ('Training', 'Test'), loc='upper left')
    plt.title('KNeighborsRegresioon')
    plt.xlabel("Label data")
    plt.ylabel("Predict-Answer")
    plt.show()

    return grid
