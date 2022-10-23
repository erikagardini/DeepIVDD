import tensorflow as tf
from python.nets.contrDA.model import resnet as model
from python.nets.contrDA.scheduler import CustomLearningRateSchedule as CustomSchedule
from python.util import load_data
from python.util.utility_func import *

net = model.__dict__['ResNet18'](
    width=1.0,
    head_dims=(512,512,512,512,512,512,512,512,128),
    input_shape=(32, 32, 3),
    num_class=2)

scheduler = CustomSchedule(
        step_per_epoch=32,
        base_lr=0.01,
        max_step=65536,
        mode='cos')

momentum = 0.9
nesterov = False
optimizer = tf.keras.optimizers.SGD(
  learning_rate=scheduler, momentum=momentum, nesterov=nesterov)

checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, model=net)

numbers = range(0,1)
pretrain_dir = "../python/nets/contrDA/contrDA_pretraining/"
seed = 0
matlab_res = 'matlab/matlab_cifar_contrDA'
data_path = "../cifar/"

for num in numbers:

    latest = tf.train.latest_checkpoint(pretrain_dir + str(num) + "/")
    checkpoint.restore(latest)

    X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

    X_red = checkpoint.model(X)["pools"]
    X_test_red = checkpoint.model(X_test)["pools"]

    X_test_red_nump = X_test_red.numpy()
    x_val1 = X_test_red_nump[np.where(y_test == 1)]
    x_val2 = X_test_red_nump[np.where(y_test == 0)]

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