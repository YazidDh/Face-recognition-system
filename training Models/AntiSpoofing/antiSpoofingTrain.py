import os


"""print(os.listdir("AntiSpoofingDataset"))

import splitfolders
splitfolders.ratio('AntiSpoofingDataset',
                   'AntiSpoofingDataset/Data',
                   seed=42,
                   ratio=(0.8, 0.2))

print()
base_dir = 'AntiSpoofingDataset/Data'
train_dir = os.path.join(base_dir, 'train') ## Train Dataset
validation_dir = os.path.join(base_dir, 'val') ## Validation Dataset


print(os.listdir('AntiSpoofingDataset/Data/train'))

print(os.listdir('AntiSpoofingDataset/Data/val'))


train_live_dir = os.path.join(train_dir, 'Live')
train_spoof_dir = os.path.join(train_dir, 'Spoof')

validation_live_dir = os.path.join(validation_dir, 'Live')
validation_spoof_dir = os.path.join(validation_dir, 'Spoof')

total_size_train = (
    len(os.listdir('AntiSpoofingDataset/Data/train/Live')) +
    len(os.listdir('AntiSpoofingDataset/Data/train/Spoof'))
)
print('Total Training Data is : {}'.format(total_size_train))

total_size_test = (
    len(os.listdir('AntiSpoofingDataset/Data/val/Live')) +
    len(os.listdir('AntiSpoofingDataset/Data/val/Spoof'))
)
print('Total Testing Data is : {}'.format(total_size_test)) """


from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


train_dir = "AntiSpoofingDataset/Data/train"
validation_dir = "AntiSpoofingDataset/Data/val"

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    fill_mode = 'nearest'
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150),
    #batch_size = 128,
    class_mode = 'binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (150, 150),
    #batch_size = 128,
    class_mode = 'binary'
)

print(train_generator)
print(validation_generator)
print(train_generator.class_mode)
print(train_generator.class_indices)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    # tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model_architecture.jpg', show_shapes=True)

model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'Adam',
    metrics = ['accuracy']
)
import tensorflow as tf

# Charger le modèle à partir du fichier .h5
model = tf.keras.models.load_model('models/AntiSpoofing_model.h5')

# Charger vos données de test
# Assurez-vous qu'elles sont dans le même format que celui sur lequel le modèle a été entraîné
# X_test contient les données de test et y_test les étiquettes de test

# Évaluer la précision du modèle
loss, accuracy = model.evaluate( validation_generator )

print("Perte (Loss):", loss)
print("Précision (Accuracy):", accuracy)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("val_accuracy") > 0.95):
            print("\nVal acc is above 94 so stoping ")
            self.model.stop_training = True


callbacks = myCallback()
history = model.fit(
    train_generator,
    epochs = 15,
    #callbacks = [reduce_LR, stop_early],
    validation_data = validation_generator,
    verbose = 1,
    callbacks=[callbacks]
)

model.save('models/AntiSpoofing_model.h5')

import matplotlib.pyplot as plt

def plot_accuracy(history):
  plt.figure(figsize=(18,5))
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  epochs = range(len(acc))
  plot_acc = plt.plot(epochs, acc, 'red', label='Training Accuracy')
  plot_val_acc = plt.plot(epochs, val_acc, 'blue', label='Validation Accuracy')
  plt.xlabel('Epoch', fontsize=15)
  plt.ylabel('Accuracy', fontsize=15)
  plt.title('Training and Validation Accuracy', fontsize=25)
  plt.legend(bbox_to_anchor=(1,1), loc='best')
  plt.grid()
  plt.show()

def plot_loss(history):
  plt.figure(figsize=(18,5))
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))
  plot_loss = plt.plot(epochs, loss, 'red', label='Training Loss')
  plot_val_loss = plt.plot(epochs, val_loss, 'blue', label='Validation Loss')
  plt.xlabel('Epoch', fontsize=15)
  plt.ylabel('Loss', fontsize=15)
  plt.title('Training and Validation Loss', fontsize=25)
  plt.legend(bbox_to_anchor=(1,1), loc='best')
  plt.grid()
  plt.show()


plot_accuracy(history)
plot_loss(history) 
