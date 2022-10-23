from python.util import load_data
import numpy as np
from python.util.utility_func import save_res
from sklearn.metrics import *

numbers = range(0,1)
seed = 0
data_path = "../cifar/"
out_dir = "../results/beta_recalibration/"
dir = '../results/res_contrDA/'

for num in numbers:

    X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

    train_scores = np.loadtxt(dir + '/IVDD-KL-train_scores' + str(num) + "_seed" + str(seed) + '.txt')
    test_scores = np.loadtxt(dir + '/IVDD-KL-test_scores' + str(num) + "_seed" + str(seed) + '.txt')

    beta_values = [1.0, 25.0, 100.0, 1000.0, 10000.0]

    f = open(out_dir + str(num) + ".txt", "w")
    for beta in beta_values:
        train_probs = 1. / (1 + np.exp(beta * train_scores))
        test_probs = 1. / (1 + np.exp(beta * test_scores))
        save_res(X, X_test, train_probs, test_probs, y_test, "ssl-IVDD_"+str(beta), f, num, out_dir+"_"+str(beta), plot_top_bottom=False)

        pred_labels = np.zeros((test_probs.shape[0]))
        pred_labels[np.where(test_probs >= 0.5)] = 1

        print(precision_score(y_test, pred_labels))
        print(recall_score(y_test, pred_labels))
        print(f1_score(y_test, pred_labels))
        print(balanced_accuracy_score(y_test, pred_labels))
        print(confusion_matrix(y_test, pred_labels))

    print("\n\n")