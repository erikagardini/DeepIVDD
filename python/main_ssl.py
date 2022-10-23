import tensorflow as tf
from sklearn.svm import OneClassSVM

from python.nets.contrDA.model import resnet as model
from python.nets.contrDA.scheduler import CustomLearningRateSchedule as CustomSchedule
from python.util import load_data
from python.ivddkl.ivddkl import IVDDKLKernel
from python.util.utility_func import *

numbers = range(0,10)
pretrain_dir = "nets/contrDA/contrDA_pretraining/"
seed = 0
out_dir = '../results/res_contrDA/'
data_path = "../cifar/"

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

for num in numbers:

    latest = tf.train.latest_checkpoint(pretrain_dir + str(num) + "/")
    checkpoint.restore(latest)

    X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

    X_red = checkpoint.model(X)["pools"]
    X_test_red = checkpoint.model(X_test)["pools"]

    X_red_norm = tf.nn.l2_normalize(X_red, axis=1)
    X_test_red_norm = tf.nn.l2_normalize(X_test_red, axis=1)

    f = open(out_dir + str(num) + "v2.txt", "w")

    # DeepIVDD Kernel KL
    ivddKL_Kernel = IVDDKLKernel(50, 1.0, beta=25.0, lossKLD=True)
    radius, inside = ivddKL_Kernel.training(X_red_norm, epochs=30, batch_size=32, lr=0.001, center_mode='kmeans',
                                     sigma_mode='ocsvm', earlystopping=True, thr_earlystopping=[0.80, 0.90])

    test_probs, scores_test = ivddKL_Kernel.testing(X_test_red_norm)
    train_probs, scores_train = ivddKL_Kernel.testing(X_red_norm)
    np.save(out_dir + "inside"+str(num), inside)
    np.save(out_dir + "radius"+str(num), radius)
    np.savetxt(out_dir + '/IVDD-KL-train_probs' + str(num) + "_seed" + str(seed) + '.txt', train_probs, delimiter=' ')
    np.savetxt(out_dir + '/IVDD-KL-test_probs' + str(num) + "_seed" + str(seed) + '.txt', test_probs, delimiter=' ')
    np.savetxt(out_dir + '/IVDD-KL-train_scores' + str(num) + "_seed" + str(seed) + '.txt', scores_train, delimiter=' ')
    np.savetxt(out_dir + '/IVDD-KL-test_scores' + str(num) + "_seed" + str(seed) + '.txt', scores_test, delimiter=' ')
    save_res(X, X_test, train_probs, test_probs, y_test, "IVDD-KL", f, num, out_dir)

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
