import matplotlib.pyplot as plt
import pandas as pd

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
  #DecisionTreeClassifier, export_graphviz
#import graphviz
###########################
def get_data():
    X = pd.read_csv('https://raw.githubusercontent.com/QuantaVox/titovox/main/ML/ParkSet2_X.csv')
    y = pd.read_csv('https://raw.githubusercontent.com/QuantaVox/titovox/main/ML/ParkSet2_y.csv')
    y = [yy[0] for yy in y.values]   # fix and save differently
    return X,y 
    
def runit(RANDOM = 42, RandomForest=True):
    X,y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM)
    if RandomForest:
        print('FITTING RandomForest')
        estimator = RandomForestClassifier(random_state=RANDOM, n_estimators=100)
    else:
        print('TPOT Classifier')
        estimator = TPOTClassifier(generations=20, cv=5, verbose=2, random_state=RANDOM)

    estimator.fit(X_train, y_train)
    return estimator, X_test, y_test


def auc_plot(tpot, X_test, y_test, plotit=False):
    y_pred_prob = tpot.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print("AUC Score:", auc_score)
    
    if plotit: # Plot the ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
    return auc_score

def arp(seed, plotit=False, use_trees=False):
    t,X,y = runit(seed, use_trees)
    auc_score = auc_plot(t,X,y,plotit)
    if use_trees:
        vizTree(t,X,y,seed)
    return t  # 
########################
def vizTree(tree, Xt, yt, seed):
    plt.figure(figsize=(20,12))
    plot_tree(tree.estimators_[0], filled=True, 
        feature_names=Xt.columns, 
        class_names= ['Healthy','Parkinson'])
    plt.show()
    plt.savefig(f'test_{seed}.png')
