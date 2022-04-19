import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, confusion_matrix
from tabulate import tabulate


def dummify(df):
    catefeats = [c for c in df.columns if df[c].dtype == 'object']
    for cf in catefeats:
        df = df.merge(pd.get_dummies(df[cf], prefix=cf), left_index=True, right_index=True)
        df = df.drop(cf, axis=1)
    return df


def get_test_roc_auc(model, folder, model_name, XX, yy, X, y):
    # train the model
    model.fit(XX, yy)
    # predict the labels
    plot_roc_curve(model, X, y)
    plt.title('ROC-AUC Plot for Model %s' % model_name)
    plt.savefig(os.path.join(folder, f"{model_name}-roc-auc.png"), dpi=128)
    # plt.show()
    

def anti_classification(attr, model, model_name, XX, yy, X, keep_prob=False, return_model=False):
    now_cols = [c for c in XX.columns if c != attr]
    XX_partial = dummify(XX[now_cols])
    model.fit(XX_partial, yy)
    X_partial = dummify(X[now_cols])
    X['Risk_pred(protected=%s, model=%s)' % (attr, model_name)] = model.predict(X_partial)
    if keep_prob:
        X['Risk_prob(protected=%s, model=%s)' % (attr, model_name)] = model.predict_proba(X_partial)[:, 1]
    if return_model:
        return X, model 
    else:
        return X


def perturb_attr(row, attr, attr_list):
    return np.random.choice([a for a in attr_list if a != row[attr]])


def print_confusion_matrix(cm):
    # part 1: print counts
    TN, FP, FN, TP = cm.ravel()
    print("------------- Confusion Matrix (Count) --------------")
    print(tabulate([
        ['Factually Good', TP, FN],
        ['Factually Bad', FP, TN]
    ], 
    headers=['', 'Predictably Good', 'Predictably Bad']))
    print("-----------------------------------------------------\n\n")

    # part 2: print counts
    N = cm.sum()
    print("------------- Confusion Matrix (Ratio) --------------")
    print(tabulate([
        ['Factually Good', f"{TP / N:.4f}", f"{FN / N:.4f}"],
        ['Factually Bad', f"{FP / N:.4f}", f"{TN / N:.4f}"]
    ], 
    headers=['', 'Predictably Good', 'Predictably Bad']))
    print("-----------------------------------------------------")

    return TP / N, FP / N, FN / N, TN / N


def get_threshold(attr, attr_val, df, prob_name, bar, attr_val2=None):
    if attr_val2 is None:
        limited = df[df[attr] == attr_val].sort_values(by=[prob_name], ascending=False)
    else:
        limited = df[(df[attr] >= attr_val) & (df[attr] < attr_val2)].sort_values(by=[prob_name], ascending=False)
    cut_off = math.ceil(len(limited) * bar)
    return limited.loc[limited.index[cut_off - 1]].at[prob_name]


def seperate_by_attr(attr, attr_val, df, true_name, pred_name, attr_val2=None):
    if attr_val2 is None:
        limited = df[df[attr] == attr_val]
    else:
        limited = df[(df[attr] >= attr_val) & (df[attr] < attr_val2)]
    return print_confusion_matrix(confusion_matrix(limited[true_name], limited[pred_name], labels=['good', 'bad']))


def boost_dataset(file_path):
    new_file_path = file_path.replace('.csv', '_augmented.csv')
    with open(new_file_path, 'w') as fw:
        for line in open(file_path, 'r'):
            # skip line
            if 'Age' in line:
                fw.write(line)
                continue
            if 'female' in line:
                fw.write(line)
                fw.write(line.replace('female', 'male'))
            else:
                fw.write(line)
                fw.write(line.replace('male', 'female'))
    return new_file_path
