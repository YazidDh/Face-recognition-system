from keras.models import load_model


from PIL import Image
from keras.models import load_model
from keras.utils import load_img
import numpy as np
import io
model = load_model('age_gen_model.h5')


def predict(image):

    gender_dict = {0: 'Male', 1: 'Female'}
    pil_image = Image.fromarray(image)

    # Convert to grayscale
    img = pil_image.convert('L')
    #img = load_img(image, color_mode="grayscale")
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img/255.0
    pred = model.predict(img)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    return pred_age, pred_gender




#age,gender = predict2("faces/MazenHaouari.jpg")
#print(predict2("faces/MazenHaouari.jpg"))