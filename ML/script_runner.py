import matplotlib.pyplot as plt
import pandas as pd

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

def runit(RANDOM = 42):
    X = pd.read_csv('https://raw.githubusercontent.com/QuantaVox/titovox/main/ML/ParkSet2_X.csv')
    y = pd.read_csv('https://raw.githubusercontent.com/QuantaVox/titovox/main/ML/ParkSet2_y.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM)

    print('FITTING RandomForest')

    tpotRFC = RandomForestClassifier(random_state=RANDOM)
    tpotRFC.fit(X_train, y_train)
    return tpotRFC, X_test, y_test

def auc_plot(tpot, X_test, y_test):
    y_pred_prob = tpot.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print("AUC Score:", auc_score)
    
    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
