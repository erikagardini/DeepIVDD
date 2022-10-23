import tensorflow as tf
from python.util import load_data
import numpy as np


numbers = range(0,10)
pretrain_dir = "../python/nets/simsiam/simsiam_pretraining/"
seed = 0
matlab_res = 'matlab/matlab_cifar_simsiam'
data_path = "../cifar/"

for num in numbers:
    model = tf.keras.models.load_model(pretrain_dir + str(num) + "/linear_model.h5")
    X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

    X_red = model(X)
    X_test_red = model(X_test)

    X_test_red = X_test_red.numpy()
    x_val1 = X_test_red[np.where(y_test == 1)]
    x_val2 = X_test_red[np.where(y_test == 0)]

    np.savetxt(matlab_res + '/x' + str(num) + "_seed" + str(seed) + '.txt', X_red, delimiter=' ')
    np.savetxt(matlab_res + '/xval1' + str(num) + "_seed" + str(seed) + '.txt', x_val1, delimiter=' ')
    np.savetxt(matlab_res + '/xval2' + str(num) + "_seed" + str(seed) + '.txt', x_val2, delimiter=' ')

    X_red_norm = tf.nn.l2_normalize(X_red, axis=1)
    X_test_red_norm = tf.nn.l2_normalize(X_test_red, axis=1)

    X_test_red_norm = X_test_red_norm.numpy()
    x_val1 = X_test_red_norm[np.where(y_test == 1)]
    x_val2 = X_test_red_norm[np.where(y_test == 0)]

    np.savetxt(matlab_res + '_norm/x' + str(num) + "_seed" + str(seed) + '.txt', X_red_norm, delimiter=' ')
    np.savetxt(matlab_res + '_norm/xval1' + str(num) + "_seed" + str(seed) + '.txt', x_val1, delimiter=' ')
    np.savetxt(matlab_res + '_norm/xval2' + str(num) + "_seed" + str(seed) + '.txt', x_val2, delimiter=' ')