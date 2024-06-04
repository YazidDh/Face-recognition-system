from PIL import Image
from keras.models import load_model
from keras.utils import load_img
import numpy as np
model = load_model('models/age_gen_model.h5')


def predict(image):
    # map labels for gender
    gender_dict = {0: 'Male', 1: 'Female'}
    img = load_img(image, color_mode="grayscale")
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img/255.0
    pred = model.predict(img)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    print(pred)
    print("Predicted Gender:", pred_gender, "\nPredicted Age: ", pred_age)


predict("test/dridi2.jpg")