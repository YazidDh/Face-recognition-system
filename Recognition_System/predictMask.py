from keras.models import load_model
import numpy as np
from PIL import Image


model = load_model('maskModel.h5')

def predictMask(image):
    image_pil = Image.fromarray(image)
    resized_image = image_pil.resize((150, 150))
    x = np.asarray(resized_image)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    return classes[0]
