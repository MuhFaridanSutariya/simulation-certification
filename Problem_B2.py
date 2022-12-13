# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc') > 0.83 and logs.get('val_acc') > 0.83):
      print("Training Accuracy and Validation Accuracy telah menyentuh!")
      self.model.stop_training = True

def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    train_images = training_images.reshape((training_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    training_images = training_images / 255
    test_images = test_images / 255

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # COMPILE MODEL HERE
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )

    # TRAIN YOUR MODEL HERE
    model.fit(
        training_images,
        training_labels,
        validation_data=(test_images, test_labels),
        epochs=100,
        callbacks=[myCallback()]
    )

    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
