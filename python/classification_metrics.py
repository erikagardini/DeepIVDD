from python.util import load_data
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

numbers = range(0,10)
seed = 0
dir = '../results/res_contrDA/'
matlab_dir = '../ivdd/matlab/probs_cifar_contrDA_norm/'
save_csv = '../results/classification_metrics/'
data_path = "../cifar/"

auc_ivdd_kl = []
auc_ivdd = []
auc_ocsvm = []
precision_ivdd_kl = []
precision_ivdd = []
precision_ocsvm = []
recall_ivdd_kl = []
recall_ivdd = []
recall_ocsvm = []
f1_ivdd_kl = []
f1_ivdd = []
f1_ocsvm = []
ba_ivdd_kl = []
ba_ivdd = []
ba_ocsvm = []

for num in numbers:

    X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

    ###IVDD
    train_probs = np.loadtxt(dir + '/IVDD-KL-train_probs' + str(num) + "_seed" + str(seed) + '.txt')
    test_probs = np.loadtxt(dir + '/IVDD-KL-test_probs' + str(num) + "_seed" + str(seed) + '.txt')

    pred_labels = np.zeros((test_probs.shape[0]))
    pred_labels[np.where(test_probs >= 0.5)] = 1

    auc_ivdd_kl.append(roc_auc_score(y_test, test_probs))
    precision_ivdd_kl.append(precision_score(y_test, pred_labels))
    recall_ivdd_kl.append(recall_score(y_test, pred_labels))
    f1_ivdd_kl.append(f1_score(y_test, pred_labels))
    ba_ivdd_kl.append(balanced_accuracy_score(y_test, pred_labels))


    ###OCSVM
    train_scores = np.loadtxt(dir + '/OC-SVM-l2-train_dist' + str(num) + "_seed" + str(seed) + '.txt')
    test_scores = np.loadtxt(dir + '/OC-SVM-l2-test_dist' + str(num) + "_seed" + str(seed) + '.txt')

    pred_labels = np.zeros((test_scores.shape[0]))
    pred_labels[np.where(test_scores > 0)] = 1

    auc_ocsvm.append(roc_auc_score(y_test, test_scores))
    precision_ocsvm.append(precision_score(y_test, pred_labels))
    recall_ocsvm.append(recall_score(y_test, pred_labels))
    f1_ocsvm.append(f1_score(y_test, pred_labels))
    ba_ocsvm.append(balanced_accuracy_score(y_test, pred_labels))


    ####IVDD
    x_val1 = X_test[np.where(y_test == 1)]
    x_val2 = X_test[np.where(y_test == 0)]
    y_val1 = np.ones((x_val1.shape[0]))
    y_val2 = np.zeros((x_val2.shape[0]))

    X_test_ivdd = np.concatenate((x_val1, x_val2))
    y_test_ivdd = np.concatenate((y_val1, y_val2))

    test_probs = np.loadtxt(matlab_dir + str(num) + "_seed" + str(seed) + "_p_test.txt")
    train_probs = np.loadtxt(matlab_dir + str(num) + "_seed" + str(seed) + "_p_train.txt")

    pred_labels = np.zeros((test_probs.shape[0]))
    pred_labels[np.where(test_probs >= 0.5)] = 1

    auc_ivdd.append(roc_auc_score(y_test_ivdd, test_probs))
    precision_ivdd.append(precision_score(y_test_ivdd, pred_labels))
    recall_ivdd.append(recall_score(y_test_ivdd, pred_labels))
    f1_ivdd.append(f1_score(y_test_ivdd, pred_labels))
    ba_ivdd.append(balanced_accuracy_score(y_test_ivdd, pred_labels))


pd.DataFrame(data={"ivdd-kl": auc_ivdd_kl, "ivdd": auc_ivdd, "ocsvm": auc_ocsvm}).to_csv(save_csv+"auc.csv")
pd.DataFrame(data={"ivdd-kl": precision_ivdd_kl, "ivdd": precision_ivdd, "ocsvm": precision_ocsvm}).to_csv(save_csv+"precision.csv")
pd.DataFrame(data={"ivdd-kl": recall_ivdd_kl, "ivdd": recall_ivdd, "ocsvm": recall_ocsvm}).to_csv(save_csv+"recall.csv")
pd.DataFrame(data={"ivdd-kl": f1_ivdd_kl, "ivdd": f1_ivdd, "ocsvm": f1_ocsvm}).to_csv(save_csv+"f1.csv")
pd.DataFrame(data={"ivdd-kl": ba_ivdd_kl, "ivdd": ba_ivdd, "ocsvm": ba_ocsvm}).to_csv(save_csv+"ba.csv")

pd.DataFrame(data={"ivdd-kl": auc_ivdd_kl, "ivdd": auc_ivdd, "ocsvm": auc_ocsvm}).mean(axis=0).to_csv(save_csv+"auc_mean.csv")
pd.DataFrame(data={"ivdd-kl": precision_ivdd_kl, "ivdd": precision_ivdd, "ocsvm": precision_ocsvm}).mean(axis=0).to_csv(save_csv+"precision_mean.csv")
pd.DataFrame(data={"ivdd-kl": recall_ivdd_kl, "ivdd": recall_ivdd, "ocsvm": recall_ocsvm}).mean(axis=0).to_csv(save_csv+"recall_mean.csv")
pd.DataFrame(data={"ivdd-kl": f1_ivdd_kl, "ivdd": f1_ivdd, "ocsvm": f1_ocsvm}).mean(axis=0).to_csv(save_csv+"f1_mean.csv")
pd.DataFrame(data={"ivdd-kl": ba_ivdd_kl, "ivdd": ba_ivdd, "ocsvm": ba_ocsvm}).mean(axis=0).to_csv(save_csv+"ba_mean.csv")



