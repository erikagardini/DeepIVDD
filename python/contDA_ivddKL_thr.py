from python.util import load_data
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import pandas as pd

numbers = range(0,10)
seed = 0
data_path = "../cifar/"
dir = '../results/res_contrDA/'
save_csv = '../results/contDA_ivddKL_thr/'
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
beta = 1000.0

for thr in thresholds:
    auc_ivdd_kl = []
    precision_ivdd_kl = []
    recall_ivdd_kl = []
    f1_ivdd_kl = []
    ba_ivdd_kl = []

    for num in numbers:

        X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

        ###IVDD
        train_scores = np.loadtxt(dir + '/IVDD-KL-train_scores' + str(num) + "_seed" + str(seed) + '.txt')
        test_scores = np.loadtxt(dir + '/IVDD-KL-test_scores' + str(num) + "_seed" + str(seed) + '.txt')

        #Recalibration with a different beta
        train_probs = 1. / (1 + np.exp(beta * train_scores))
        test_probs = 1. / (1 + np.exp(beta * test_scores))

        pred_labels = np.zeros((test_probs.shape[0]))
        pred_labels[np.where(test_probs >= thr)] = 1

        auc_ivdd_kl.append(roc_auc_score(y_test, test_probs))
        precision_ivdd_kl.append(precision_score(y_test, pred_labels))
        recall_ivdd_kl.append(recall_score(y_test, pred_labels))
        f1_ivdd_kl.append(f1_score(y_test, pred_labels))
        ba_ivdd_kl.append(balanced_accuracy_score(y_test, pred_labels))

    pd.DataFrame(data={"ivdd-kl": auc_ivdd_kl}).to_csv(save_csv+str(thr)+"auc.csv")
    pd.DataFrame(data={"ivdd-kl": precision_ivdd_kl}).to_csv(save_csv+str(thr)+"precision.csv")
    pd.DataFrame(data={"ivdd-kl": recall_ivdd_kl}).to_csv(save_csv+str(thr)+"recall.csv")
    pd.DataFrame(data={"ivdd-kl": f1_ivdd_kl}).to_csv(save_csv+str(thr)+"f1.csv")
    pd.DataFrame(data={"ivdd-kl": ba_ivdd_kl}).to_csv(save_csv+str(thr)+"ba.csv")

    pd.DataFrame(data={"ivdd-kl": auc_ivdd_kl}).mean(axis=0).to_csv(save_csv+str(thr)+"auc_mean.csv")
    pd.DataFrame(data={"ivdd-kl": precision_ivdd_kl}).mean(axis=0).to_csv(save_csv+str(thr)+"precision_mean.csv")
    pd.DataFrame(data={"ivdd-kl": recall_ivdd_kl}).mean(axis=0).to_csv(save_csv+str(thr)+"recall_mean.csv")
    pd.DataFrame(data={"ivdd-kl": f1_ivdd_kl}).mean(axis=0).to_csv(save_csv+str(thr)+"f1_mean.csv")
    pd.DataFrame(data={"ivdd-kl": ba_ivdd_kl}).mean(axis=0).to_csv(save_csv+str(thr)+"ba_mean.csv")