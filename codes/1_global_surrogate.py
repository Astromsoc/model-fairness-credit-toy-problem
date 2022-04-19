# utilities
import os
import pandas as pd
import matplotlib.pyplot as plt

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# constants
from const import *

# udfs
from utils import *


if __name__ == '__main__':
    
    # load datasets
    labeled_df_raw = pd.read_csv(LABELED_DATA_PATH)
    # test_df_raw = pd.read_csv(UNLABELED_DATA_PATH)

    # get inputs & labels
    labeled_y, labeled_X = labeled_df_raw['Risk'],  dummify(labeled_df_raw.drop('Risk', axis=1))
    train_X, val_X, train_y, val_y = train_test_split(labeled_X, labeled_y, test_size=VAL_RATIO, random_state=RANDOM_SEED)

    # initiate the models
    models = [
        (LogisticRegression(), 'LR'),
        (RandomForestClassifier(), 'RF'),
        (LinearDiscriminantAnalysis(), 'LDA')
    ]

    results_roc = []
    names_roc = []
    scoring_roc = 'roc_auc'

    # train the models
    for model, name in models:
        kfold = KFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)
        cv_results = cross_val_score(model, train_X, train_y, cv=kfold, scoring=scoring_roc)
        results_roc.append(cv_results)
        names_roc.append(name)
    
    # boxplot algorithm comparison
    fig = plt.figure(figsize=(12,5))
    fig.suptitle('Algorithm Comparison: with ROC-AUC')
    ax = fig.add_subplot(111)
    plt.boxplot(results_roc)
    ax.set_xticklabels(names_roc)
    plt.savefig(os.path.join(RESULTS_PATH, 'model-comparison.png'), dpi=128)
    # plt.show()

    # use all training data without CV and evaluate on validation with ROC-AUC
    for model, name in models:
        get_test_roc_auc(model, RESULTS_PATH, name, train_X, train_y, val_X, val_y)
