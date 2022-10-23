import tensorflow as tf
from python.util import load_data
from sklearn.svm import OneClassSVM
from python.util.utility_func import *


numbers = range(0,10)
pretrain_dir = "nets/simsiam/simsiam_pretraining/"
seed = 0
out_dir = '../results/res_simsiam/'
data_path = "../cifar/"

for num in numbers:
    model = tf.keras.models.load_model(pretrain_dir + str(num) + "/linear_model.h5")
    X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

    X_red = model(X)
    X_test_red = model(X_test)

    X_red_norm = tf.nn.l2_normalize(X_red, axis=1)
    X_test_red_norm = tf.nn.l2_normalize(X_test_red, axis=1)

    f = open(out_dir + str(num) + ".txt", "w")

    nu = 0.5
    kernel = 'rbf'
    # OCSVM kernel
    gamma = 10. / (tf.math.reduce_variance(X_red_norm) * X_red_norm.shape[1]).numpy()
    clf = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu).fit(X_red_norm)
    scores_test = clf.score_samples(X_test_red_norm)
    scores_train = clf.score_samples(X_red_norm)
    save_res(X, X_test, scores_train, scores_test, y_test, "OCSVM-l2", f, num, out_dir)

    np.savetxt(out_dir + '/OC-SVM-l2-test_scores' + str(num) + "_seed" + str(seed) + '.txt', scores_test, delimiter=' ')
    np.savetxt(out_dir + '/OC-SVM-l2-train_scores' + str(num) + "_seed" + str(seed) + '.txt', scores_train, delimiter=' ')

    dist_test = clf.decision_function(X_test_red_norm)
    dist_train = clf.decision_function(X_red_norm)
    save_res(X, X_test, dist_train, dist_test, y_test, "OCSVM-l2-dist", f, num, out_dir)

    np.savetxt(out_dir + '/OC-SVM-l2-test_dist' + str(num) + "_seed" + str(seed) + '.txt', dist_test, delimiter=' ')
    np.savetxt(out_dir + '/OC-SVM-l2-train_dist' + str(num) + "_seed" + str(seed) + '.txt', dist_train, delimiter=' ')

    # OCSVM kernel
    gamma = 10. / (tf.math.reduce_variance(X_red) * X_red.shape[1]).numpy()
    clf = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu).fit(X_red)
    scores_test = clf.score_samples(X_test_red)
    scores_train = clf.score_samples(X_red)
    save_res(X, X_test, scores_train, scores_test, y_test, "OCSVM", f, num, out_dir)

    np.savetxt(out_dir + '/OC-SVM-test_scores' + str(num) + "_seed" + str(seed) + '.txt', scores_test, delimiter=' ')
    np.savetxt(out_dir + '/OC-SVM-train_scores' + str(num) + "_seed" + str(seed) + '.txt', scores_train, delimiter=' ')

    dist_test = clf.decision_function(X_test_red)
    dist_train = clf.decision_function(X_red)
    save_res(X, X_test, dist_train, dist_test, y_test, "OCSVM-dist", f, num, out_dir)

    np.savetxt(out_dir + '/OC-SVM-test_dist' + str(num) + "_seed" + str(seed) + '.txt', dist_test, delimiter=' ')
    np.savetxt(out_dir + '/OC-SVM-train_dist' + str(num) + "_seed" + str(seed) + '.txt', dist_train, delimiter=' ')