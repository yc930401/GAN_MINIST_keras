import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.models import load_model

generator_epochs = 15
discriminator_epochs = 4
data_size = 256
batch_size = 128
iterations = 500
dropout_prob = 0.5

def discriminator():
    model = Sequential()
    input_shape = (28, 28, 1)

    model.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
    model.add(LeakyReLU())

    model.add(Conv2D(128, 5, strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_prob))

    model.add(Conv2D(256, 5, strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_prob))

    model.add(Conv2D(512, 5, strides=1, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_prob))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def generator():
    model = Sequential()
    model.add(Dense(7 * 7 * 256, input_dim=100))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 256)))
    model.add(Dropout(dropout_prob))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(64, 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU())

    model.add(Conv2D(32, 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU())

    model.add(Conv2D(1, 5, padding='same'))
    model.add(Activation('sigmoid'))

    return model


def get_models(discriminator, generator):
    optim_discriminator = Adam(lr=0.0001, clipvalue=1.0, decay=1e-10) # RMSprop lr=0.0008
    model_discriminator = Sequential()
    model_discriminator.add(discriminator)
    model_discriminator.compile(loss='binary_crossentropy', optimizer=optim_discriminator, metrics=['accuracy'])
    model_discriminator.summary()

    optim_adversarial = Adam(lr=0.0002, clipvalue=1.0, decay=1e-10)
    model_adversarial = Sequential()
    model_adversarial.add(generator)
    # Disable layers in discriminator
    for layer in discriminator.layers:
        layer.trainable = False
    model_adversarial.add(discriminator)
    model_adversarial.compile(loss='binary_crossentropy', optimizer=optim_adversarial, metrics=['accuracy'])
    model_adversarial.summary()
    return model_discriminator, model_adversarial


def get_whole_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test


def get_real_fake_data(generator, x, data_size):
    # Select a random set of training images from the mnist dataset
    images_train = x[np.random.randint(0, x.shape[0], size=data_size), :, :, :]
    # Generate a random noise vector
    noise = np.random.uniform(-1.0, 1.0, size=[data_size, 100])
    # Use the generator to create fake images from the noise vector
    images_fake = generator.predict(noise)

    # Create a dataset with fake and real images
    x = np.concatenate((images_train, images_fake))
    y = np.ones([2 * data_size, 1])
    y[data_size:, :] = 0
    return x, y


def plot_images(path, images):
    plt.figure(figsize=(15, 6))
    for i in range(40):
        image = images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.subplot(4, 10, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path + '.png')
    #plt.show()


if __name__ == '__main__':
    if os.path.exists('discriminator.h5'):
        discriminator = load_model('discriminator.h5')
    else:
        discriminator = discriminator()
    if os.path.exists('generator.h5'):
        generator = load_model('generator.h5')
    else:
        generator = generator()
    discriminator.summary()
    generator.summary()

    x_train, y_train, x_test, y_test = get_whole_data()
    model_discriminator, model_adversarial = get_models(discriminator, generator)

    loss_adv = []
    loss_dis = []
    acc_adv = []
    acc_dis = []
    plot_iteration = []

    for i in range(1, iterations + 1):
        print('================== Iteration {} ==================='.format(i))
        discriminator_epochs = min(30, discriminator_epochs+1)
        x, y = get_real_fake_data(generator, x_train, data_size)
        x_, y_ = get_real_fake_data(generator, x_test, data_size)

        # Train discriminator for one batch
        print('------------------ Train discriminator--------------------')
        d_stats = model_discriminator.fit(x, y, epochs=discriminator_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_data=(x_, y_))

        # Train the generator
        # The input of th adversarial model is a list of noise vectors. The generator is 'good' if the discriminator classifies
        # all the generated images as real. Therefore, the desired output is a list of all ones.
        y = np.ones([data_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[data_size, 100])
        noise_ = np.random.uniform(-1.0, 1.0, size=[data_size, 100])
        print('-------------------- Train generator----------------------')
        a_stats = model_adversarial.fit(noise, y, epochs=generator_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_data=(noise_, y))
        generator.save('generator.h5')
        discriminator.save('discriminator.h5')

        print(d_stats.history, a_stats.history)

        # Plot real and fake images
        plot_images('Real_images', x_train[np.random.randint(0, x_train.shape[0], size=40), :, :, :])
        plot_images('Fake_images', generator.predict(np.random.uniform(-1.0, 1.0, size=[40, 100])))

        # plot loss and acc
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)

        ax1.plot(a_stats.history['loss'], label='adversarial loss')
        ax1.plot(d_stats.history['loss'], label='discriminator loss')
        ax1.legend()

        ax2.plot(a_stats.history['acc'], label='adversarial acc')
        ax2.plot(d_stats.history['acc'], label='discriminator acc')
        ax2.legend()
        plt.savefig('Acc_Loss.png')
        #plt.show()

        plt.close('all')