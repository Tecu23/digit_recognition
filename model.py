import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Model:
    def __init__(self):
        self.model = None

    def load(self):
        self.model =  tf.keras.models.load_model('digit_recognition.model')

    def create_simple_model(self):
        # Creating the model
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))

    def create_convolutional_model(self,num_classes=10):
        # Creating the model
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    def train(self, train_data, x_test, y_test, batch_size=128, epochs=10):
        # Compiling the model
        self.model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

        # Training the model
        self.model.fit(train_data[0], train_data[1],batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

        # Saving the model
        self.model.save('digit_recognition.model')

    def test(self, test_data):
        loss, accuracy = self.model.evaluate(test_data[0], test_data[1])

        return loss, accuracy

    def verify(self):
        images = ['unu.png', 'doi.png', 'trei.png', 'patru.png', 'cinci.png', 'sase.png', 'sapte.png', 'opt.png', 'noua.png', 'zero.png']
        
        try:
            for image in images:
                img = cv2.imread(f"digits/{image}")[:,:,0]
                img = np.invert(np.array([img]))
                prediction = self.model.predict(img)
                # print(prediction)
                print(f"The number is probably a {np.argmax(prediction)}")
                plt.imshow(img[0], cmap=plt.cm.binary)
                plt.show()
        except:
            print("Error!")