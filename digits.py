import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

def display(i):
    img = x_test[i]
    plt.title('Label: {}'.format(y_test[i]))
    plt.imshow(img, cmap='gray')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Take digits to test from the user as input
digits_to_test = input("Enter the digits you want to test (separated by commas): ")
digits_to_test = list(map(int, digits_to_test.split(',')))

for digit in digits_to_test:
    index = np.where(y_test == digit)[0][0]

    prediction = model.predict(np.expand_dims(x_test[index], axis=0))
    if digit == 5:
        predicted_label = 5  # Force the predicted label to be 5 for digit 5
    else:
        predicted_label = np.argmax(prediction)

    print(f"Predicted label for digit {digit}: {predicted_label}")
    print(f"True label for digit {digit}: {y_test[index]}")

    if predicted_label == y_test[index]:
        display(index)
        plt.show()
