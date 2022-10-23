from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, \
    Conv2DTranspose, Layer, LeakyReLU, AveragePooling2D, Activation, BatchNormalization
from tensorflow import keras
import tensorflow as tf


#return the model architecture according to the desired experiment
def getModel():
    #Encoder
    encoder_inputs = keras.Input(shape=(32, 32, 3))

    x = Conv2D(32, 5, activation=None, strides=2, padding="same")(encoder_inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, 5, activation=None, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, 3, activation=None, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    z_mean = Dense(128, name="z_mean")(x)
    z_log_var = Dense(128, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    #Decoder
    latent_inputs = keras.Input(shape=(128,))
    x = Dense(4 * 4 * 128, activation="relu")(latent_inputs)
    x = Reshape((4, 4, 128))(x)
    x = Conv2DTranspose(128, 5, activation=None, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2DTranspose(64, 5, activation=None, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2DTranspose(32, 5, activation=None, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    decoder_outputs = Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    outputs = decoder(encoder(encoder_inputs)[2])
    model = keras.Model(encoder_inputs, outputs, name='vae_mlp')

    return model, encoder, decoder


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(data, reconstruction))
            #reconstruction_loss = tf.reduce_mean(
            #    keras.losses.binary_crossentropy(data, reconstruction)
            #)
            reconstruction_loss *= data.shape[1] * data.shape[2]
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }