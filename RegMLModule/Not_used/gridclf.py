"""
grid and hyper parameter auto search program
including heatmap plot
Lasso Disissiontree, Randomforest, Gradientboosting
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')
import scipy as sp

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

def grid_logis(X_train, y_train, X_test, y_test, X_name_list, cl_name_list):
    '''
     GridSearch lasso
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list: namelist
    :
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))
    cl = list(range(0, len(cl_name_list)))
    # 正則化パラメータ
    Logisparams = {'C': [1, 10, 100]}

    # Crossvaridationを5回、すべてのコアを使う
    grid = GridSearchCV(LogisticRegression(), param_grid=Logisparams, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # CVを制御したいとき　クラス分類ではStratifyKFold,回帰ではKFold
    # from sklearn.model_selection import KFold
    # GridSearchCVの引数に　cv=KFold(n_splits=5, shuffle=True, random_state=1)

    # R2スコアー
    print("LogisticRegression")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))

    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    # print("Number of features used: {}".format(np.sum(grid.coef_ != 0)))

    #
    clf_train = grid.predict(X_train)
    clf_test = grid.predict(X_test)

    print("Accuracy:{:.3f}".format(accuracy_score(y_test, clf_test)))

    # Confusion matrix
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test, clf_test)))

    # heatmap
    print(confusion_matrix(y_test, clf_test))

    # 各クラスの適合率（precision）、再現率（recall）、F1スコア（F1-score）、支持度(サポート)を表形式で表示
    print(classification_report(y_test, clf_test))

    # heatmap
    plt.matshow(confusion_matrix(y_test, clf_test))
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.xticks(cl, cl_name_list)
    plt.ylabel("True label")
    plt.yticks(cl, cl_name_list)
    plt.show()


def grid_decisiontreeclf(X_train, y_train, X_test, y_test, X_name_list, cl_name_list):
    '''

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param Xl:
    :param X_name:
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))
    cl = list(range(0, len(cl_name_list)))

    # 正則化パラメータ
    DTparams = {'max_depth': [2, 3, 4, 5]}

    # Crossvaridationを5回、すべてのコアを使う
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=DTparams, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)


    # R2スコアー
    print("DecisionTree")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))

    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    # print("Number of features used: {}".format(np.sum(grid.coef_ != 0)))

    #
    clf_train = grid.predict(X_train)
    clf_test = grid.predict(X_test)

    print("Accuracy:{:.3f}".format(accuracy_score(y_test, clf_test)))

    # Confusion matrix
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test, clf_test)))

    # heatmap
    print(confusion_matrix(y_test, clf_test))

    # 各クラスの適合率（precision）、再現率（recall）、F1スコア（F1-score）、支持度(サポート)を表形式で表示
    print(classification_report(y_test, clf_test))

    # heatmap
    plt.matshow(confusion_matrix(y_test, clf_test))
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.xticks(cl, cl_name_list)
    plt.ylabel("True label")
    plt.yticks(cl, cl_name_list)
    plt.show()

    # 決定木の特徴量の重要性
    # GridSearchCVはfeature_importances_メソッド持っていないためベストの値でインスタンスを生成する。
    clf = grid.best_estimator_
    clf.fit(X_train, y_train)
    print("Feature importances:\n{}".format(clf.feature_importances_))
    plt.plot(clf.feature_importances_, 's')
    plt.title("DecisionTree feature importances")
    plt.xlabel("Label data")
    plt.ylabel("Importance")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

def grid_randomforestclf(X_train,y_train,X_test,y_test,X_name_list, cl_name_list):
    '''
     GridSearch RandomForest
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param Xl:
    :param X_name:
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))
    cl = list(range(0, len(cl_name_list)))


    RFparams = {'n_estimators': [5, 10, 50, 100, 200]}
    # estimatorsは大きければ大きい程よいが、より多くの決定木の平均値をとり過剰適合が低減される。訓練時間が多くかかることが問題。
    grid = GridSearchCV(RandomForestClassifier(), param_grid=RFparams, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)


    # R2スコアー
    print("RandomForest")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))

    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    # print("Number of features used: {}".format(np.sum(grid.coef_ != 0)))

    clf_train = grid.predict(X_train)
    clf_test = grid.predict(X_test)

    print("Accuracy:{:.3f}".format(accuracy_score(y_test, clf_test)))

    # Confusion matrix
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test, clf_test)))

    # heatmap
    print(confusion_matrix(y_test, clf_test))

    # 各クラスの適合率（precision）、再現率（recall）、F1スコア（F1-score）、支持度(サポート)を表形式で表示
    print(classification_report(y_test, clf_test))

    # heatmap
    plt.matshow(confusion_matrix(y_test, clf_test))
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.xticks(cl, cl_name_list)
    plt.ylabel("True label")
    plt.yticks(cl, cl_name_list)
    plt.show()

    # RandomForestの特徴量の重要性
    clf = grid.best_estimator_
    clf.fit(X_train, y_train)
    print("Feature importances:\n{}".format(clf.feature_importances_))
    plt.plot(clf.feature_importances_, 's')
    plt.title("RandomForest feature importances")
    plt.xlabel("Label data")
    plt.ylabel("Importance")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()


def grid_gadientboosting(X_train,y_train,X_test,y_test,X_name_list, cl_name_list):
    '''
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name:
    :return:
    '''
    # GradientBoosting
    Xl = list(range(0, len(X_name_list)))
    cl = list(range(0, len(cl_name_list)))


    GBparams = {'max_depth': [1, 2, 3, 4, 5], 'n_estimators': [5, 10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

    grid = GridSearchCV(GradientBoostingClassifier(), param_grid=GBparams, cv=5, n_jobs=-1)

    grid.fit(X_train, y_train)

    # R2スコアー
    print("GradientBoosting")
    print("Training Best score : {}".format(grid.score(X_train, y_train)))

    print("Test Best score : {}".format(grid.score(X_test, y_test)))
    print("Best paramator: {}".format(grid.best_params_))
    # print("Number of features used: {}".format(np.sum(grid.coef_ != 0)))

    clf_train = grid.predict(X_train)
    clf_test = grid.predict(X_test)

    print("Accuracy:{:.3f}".format(accuracy_score(y_test, clf_test)))

    # Confusion matrix
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test, clf_test)))

    # heatmap
    print(confusion_matrix(y_test, clf_test))

    # 各クラスの適合率（precision）、再現率（recall）、F1スコア（F1-score）、支持度(サポート)を表形式で表示
    print(classification_report(y_test, clf_test))

    # heatmap
    plt.matshow(confusion_matrix(y_test, clf_test))
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.xticks(cl, cl_name_list)
    plt.ylabel("True label")
    plt.yticks(cl, cl_name_list)
    plt.show()

    # RandomForestの特徴量の重要性
    clf = grid.best_estimator_
    clf.fit(X_train, y_train)
    print("Feature importances:\n{}".format(clf.feature_importances_))
    plt.plot(clf.feature_importances_, 's')
    plt.title("GradientBoosting feature importances")
    plt.xlabel("Label data")
    plt.ylabel("Importance")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()