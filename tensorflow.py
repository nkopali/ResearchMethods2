import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

def main():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Preprocess data
    x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train)
    x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # Adapt the model for CIFAR-10
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Run inference on some test images
    predictions = model.predict(x_test[:4])
    print("Predictions:", predictions.argmax(axis=1))

if __name__ == '__main__':
    main()
