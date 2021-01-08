##### Model Test #####
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import sys

IMAGE_SIZE = 128
categories = {0: "cat", 1: "dog"}

SAMPLE_IMAG_PATH = "samples\\img_test.jpg"

def main():

    args = sys.argv
    image_path = SAMPLE_IMAG_PATH

    if (len(args) > 1):
        image_path = args[1]

    print("Loading ", image_path)
    try:
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    except :
        print("Error loading ", image_path)
        return

    print("Loading Model...")
    try:
        model = load_model("cat_dog_model.h5")
    except :
        print("Error loading model")
        return
    
    # # convert to array
    img = img_to_array(img)
    # # reshape into a single sample with 3 channels
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    # # normalize pixels
    img = img.astype('float32')
    img = img / 255.0

    # # make prediction
    print("Processing", image_path, "\n")
    res = (model.predict(img, verbose = 0) > 0.5).astype("int32")   

    print("Predicted Class : ", str(categories.get(res.flatten()[0])))


if __name__ == "__main__":
    main()
