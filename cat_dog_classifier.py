from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import keras.regularizers

import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img


train_dir = "cat_dog_dataset\\training_set"
test_dir = "cat_dog_dataset\\test_set"

IMAGE_SIZE = 128
BATCH_SIZE = 32


def plot_samples():
    dog_folder = train_dir + "\\dogs\\"
    cat_folder = train_dir + '\\cats\\'
    
    for i in range(2):
        plt.subplot(2,2,i + 1)
        # define filename
        filename = dog_folder + 'dog.' + str(i+1) + '.jpg'
        # load image pixels
        image = load_img(filename)
        # plot raw pixel data
        plt.imshow(image)

    for j in range(2):
        plt.subplot(2,2,2+j + 1)
        # define filename
        filename = cat_folder + 'cat.' + str(j+1) + '.jpg'
        # load image pixels
        image = load_img(filename)
        # plot raw pixel data
        plt.imshow(image)

    plt.show()


def summarize_diagnostics(history):
	# plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.legend()
    plt.show()


def load_data() :

    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                       width_shift_range=0.1, 
                                       height_shift_range=0.1, 
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'binary')

    test_set = test_datagen.flow_from_directory(test_dir,
                                                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'binary')
    
    return [training_set, test_set]


def get_model() :

    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), activation = 'relu'))

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer=keras.regularizers.l1(1e-5)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_regularizer=keras.regularizers.l1(1e-5)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    model.add(Dense(units = 128, activation = 'relu', kernel_regularizer=keras.regularizers.l1(1e-5)))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the CNN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


from keras.models import load_model

model = load_model("cat_dog_model.h5")

dataset = load_data()
train_set = dataset[0]
test_set = dataset[1]

print("Categories:")
for k in test_set.class_indices.keys():
    print(" ",k)

    
print("\n\n", model.summary(), "\n")

print("Evaluating model...")
loss, acc = model.evaluate(test_set, steps=len(test_set))

print("Model loss --> ", str(loss))
print("Model accuracy --> ", str(acc))

