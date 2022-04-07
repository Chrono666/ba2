from model.custom_model import load_model
from tensorflow.keras.models import load_model

#model1, metadata = load_model('./saved_models/vgg16_2022-04-06-19-11-35.hdf5')
model = load_model('saved_models')


model.predict()
