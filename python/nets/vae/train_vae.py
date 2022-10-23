from python.nets.vae.architecture import getModel, VAE
from tensorflow import keras
import matplotlib.pyplot as plt
from python.util import load_data

filename = "vae_pretraining/"
data_path = "../../../cifar/"
epochs = 150
batch_size = 200
lr = 0.001
numbers = range(0, 10)
seed = 0
for num in numbers:
    X, y, X_test, y_test = load_data.getCifarData(num, seed, data_path=data_path)

    _, encoder, decoder = getModel()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(lr=lr))

    history = vae.fit(X, epochs=epochs, batch_size=batch_size, shuffle=True)
    vae.save_weights(filename + str(num) + "/pretraining.h5")

    plt.xticks(range(1, epochs+1))
    plt.plot(history.history.get('loss'), marker='o', label='training_loss')
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.grid()
    plt.legend()
    plt.savefig(filename + str(num) + "/pretraining_loss.png")
    plt.close()
