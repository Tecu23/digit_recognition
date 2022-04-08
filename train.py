from pickletools import optimize
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import Model

def load_data():
    # Loading data and splitting it in train and test subsets
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizing the pixels for easier
    # x_train = tf.keras.utils.normalize(x_train, axis=1)
    # x_test = tf.keras.utils.normalize(x_test, axis=1)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)

def main():

    (x_train, y_train), (x_test, y_test) = load_data()

    model = Model()

    # model.create_convolutional_model()
    # model.train((x_train, y_train), x_test, y_test)

    # loss, accuracy = model.test((x_test, y_test))

    model.load()

    # print(f"Loss is {loss} and accuracy is {accuracy}")

    model.verify()

if __name__ == '__main__':
    main()