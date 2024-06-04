from keras.models import load_model


model = load_model('models/AntiSpoofing_model.h5')

import numpy as np

from keras.utils import load_img



#1 : spoof
#0 : no spoof
img = load_img("testData/tst2s.png", target_size=(150, 150))
print(img.size)
x = np.asarray(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images)
print(classes)