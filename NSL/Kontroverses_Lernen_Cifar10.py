from cifar10_NN import *
from tensorflow.keras import datasets, layers, models, preprocessing
import neural_structured_learning as nsl 

################################################
# Im Folgenden wird auf die Tensorflow-Tutorien verwiesen (Link): 
# https://github.com/tensorflow/neural-structured-learning/tree/master/g3doc/tutorials

number_of_epochs=10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
mean = np.mean(train_images)
std = np.std(train_images)
test_images = (test_images - mean) / std
train_images = (train_images - mean) / std
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
base_model = create_and_get_NN(number_of_epochs) 

# Methoden und Variable um richtige Formatierung der zusätzlichen Daten der Wrapper-Klasse
# übergeben zu können
IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'
def convert_to_dictionaries(image, label):
  return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}

train_set_for_adv_model = convert_to_dictionaries(train_images,train_labels)

# Konfigurationsobjekt
adv_config = nsl.configs.make_adv_reg_config(
        multiplier=0.2,
        adv_step_size=0.2,
        adv_grad_norm='infinity')
# Erstellen des Regulierungs-Modells
adv_model = nsl.keras.AdversarialRegularization(base_model, label_keys=[LABEL_INPUT_NAME], adv_config = adv_config ) 
# Kompilieren und Trainieren des Modells wie bei Basis-Modell 
adv_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
adv_model.fit(train_set_for_adv_model, epochs=number_of_epochs)