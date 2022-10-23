import tensorflow as tf
from python.nets.vae.architecture import getModel
from python.util import load_data
import numpy as np

numbers = range(0,10)
pretrain_dir = "../python/nets/vae/vae_pretraining/"
seed = 0
matlab_res = 'matlab/matlab_cifar_vae'
data_path = "../cifar/"

for num in numbers:
    model, encoder, decoder = getModel()
    model.load_weights(pretrain_dir + str(num) + "/pretraining.h5")
    X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

    _, _, X_red = encoder(X)
    _, _, X_test_red = encoder(X_test)

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