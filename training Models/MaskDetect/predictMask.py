from keras.models import load_model


model = load_model('models/maskModel.h5')

import numpy as np

from keras.utils import load_img

def predict(image):
    img = load_img(image, target_size=(150,150))

    x = np.asarray(img)
    x = np.expand_dims(x, axis=0)


    #with mask 0
    #without mask 1
    images = np.vstack([x])
    classes = model.predict(images)

    return classes[0]

print(predict("testimages/yazid003.jpg"))