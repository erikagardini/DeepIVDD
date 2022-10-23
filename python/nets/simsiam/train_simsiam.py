import os
from python.util import load_data
from python.nets.simsiam.architecture import *
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 256
EPOCHS = 1
CROP_TO = 32

data_path="../../../cifar/"
out_dir = "simsiam_pretraining/"
numbers = range(0, 10)
seed = 0

for num in numbers:

    #Load data
    [x_train, y_train, x_test, y_test] = load_data.getCifarData(num, seed, data_path=data_path, normalize=False)

    ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_one = (
        ssl_ds_one.shuffle(x_train.shape[0], seed=123)
            .map(custom_augment, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_two = (
        ssl_ds_two.shuffle(x_train.shape[0], seed=123)
            .map(custom_augment, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    # We then zip both of these datasets.
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

    # Visualize a few augmented images.
    sample_images_one = next(iter(ssl_ds_one))
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(sample_images_one[n].numpy().astype("int"))
        plt.axis("off")
    #plt.show()
    plt.savefig(out_dir + str(num) +"/1.png")
    plt.close()

    # Ensure that the different versions of the dataset actually contain
    # identical images.
    sample_images_two = next(iter(ssl_ds_two))
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(sample_images_two[n].numpy().astype("int"))
        plt.axis("off")
    #plt.show()
    plt.savefig(out_dir + str(num) + "/2.png")
    plt.close()

    # Create a cosine decay learning scheduler.
    num_training_samples = len(x_train)

    steps = EPOCHS * (num_training_samples // BATCH_SIZE)
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1, decay_steps=steps
    )

    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )

    # Compile model and start training.
    simsiam = SimSiam(get_encoder(), get_predictor())
    simsiam.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
    history = simsiam.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])

    # Visualize the training progress of the model.
    plt.plot(history.history["loss"])
    plt.grid()
    plt.title("Negative Cosine Similairty")
    plt.savefig(out_dir + str(num) + "/ncs.png")
    plt.close()

    simsiam.save_model(out_dir + str(num) +"/")

    # Extract the backbone ResNet20.
    backbone = tf.keras.Model(
        simsiam.encoder.input, simsiam.encoder.get_layer("backbone_pool").output
    )

    # We then create our linear classifier and train it.
    backbone.trainable = False
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    outputs = backbone(inputs, training=False)
    linear_model = tf.keras.Model(inputs, outputs, name="linear_model")
    linear_model.save(out_dir + str(num) + '/linear_model.h5')