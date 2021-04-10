import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import numpy as np
import tensorflow.keras as k
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
import matplotlib.pyplot as plt
from Softmax_with_Temp import * 
from datetime import datetime

########################################################
# Neural network (NN)
# Initalizes and creates NN with a Softmax Layer that takes a parameter "Temperature"
def create_NN_with_temp(number_of_epochs = 10, show_model_architecture = True, show_grafics = False, show_train_eval = True, verbose = 1, model_save_path = '', temperature = 1, dataset = datasets.cifar10, class_names = [], with_softlabels = False, softlabels = None):
    # Loading images
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    #Formatting
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
    test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    train_labels = k.utils.to_categorical(train_labels, 10)
    test_labels = k.utils.to_categorical(test_labels, 10)

    mean = np.mean(train_images)
    std = np.std(train_images)
    test_images = (test_images - mean) / std
    train_images = (train_images - mean) / std

    train_labels = train_labels.astype('int')
    test_labels = test_labels.astype('int')


    if (show_grafics):
       plt.figure(figsize=(10,10))
       for i in range(25):
           plt.subplot(5,5,i + 1)
           plt.xticks([])
           plt.yticks([])
           plt.grid(False)
           plt.imshow(train_images[i], cmap=plt.cm.binary)
           #The CIFAR labels happen to be arrays, which is
           #why you need the
           #extra index
           train_labels = train_labels.astype('int')
           plt.xlabel(class_names[train_labels[i][0]])
       plt.show()      

    # Getting the model
    model = define_model(show_model_architecture, temperature = temperature)

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=15,
      horizontal_flip=True,
      width_shift_range=0.1,
      height_shift_range=0.1,
      zoom_range=0.3)
    datagen.fit(train_images)

    # Training
    if (with_softlabels): 
        history = model.fit(datagen.flow(train_images, softlabels, batch_size=128),
                            steps_per_epoch = len(train_images) / 128, 
                            epochs=number_of_epochs,
                            validation_data=(test_images, test_labels),
                            verbose = 1)
    else: 
        history = model.fit(datagen.flow(train_images, train_labels, batch_size=128),
                            steps_per_epoch = len(train_images) / 128, 
                            epochs=number_of_epochs,
                            validation_data=(test_images, test_labels),
                            verbose = 1)
    plothist(history)
    print('Model training is done!')

    if (show_grafics):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

    if (model_save_path != ''):    
        model.save(model_save_path)
        print('Model is saved under: ' , model_save_path)
    else:
        print('Model is not saved...')
    
    return model

def plothist (hist):
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Defines the model architecture (layers, number of layers, metrics,...)
def define_model(showModelArchitecture, temperature):
    #make model
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32, kernel_size=3 , activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))   # reduces to 16x16x3xnum_filters
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))   # reduces to 8x8x3x(2*num_filters)
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))   # reduces to 4x4x3x(4*num_filters)
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(10))

    #Custom Softmax Layer
    model.add(Softmax_with_Temp(temperature=temperature))

    if (showModelArchitecture):
        model.summary()
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=0.001, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    return model

# Load saved Model
def load_saved_Model (path, show_summary = False):
    model = models.load_model(path)
    if (show_summary):
        model.summary()
    print(model.loss)
    return model

#Creating Soft Labels by predicting with the teacher model
def create_softlabels(dataset = datasets.cifar10, savepath = "", model = load_saved_Model('./models/model_1_25epochs')):
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)

    train_images = train_images.astype('float32')

    train_labels = k.utils.to_categorical(train_labels, 10)

    mean = np.mean(train_images)
    std = np.std(train_images)
    train_images = (train_images - mean) / std

    train_labels = train_labels.astype('int')

    i = len(train_labels)
    n = 0
    train_soft_labels = []

    while n < i:
        img = (np.expand_dims(train_images[n], 0))
        x = model.predict(img)  
        train_soft_labels.append(x)
        print(n)
        n=n+1
    train_soft_labels = np.array(train_soft_labels)
    np.save(savepath, train_soft_labels)


    train_soft_labels_reshaped = train_soft_labels.reshape(train_soft_labels.shape[0], -1)

    print("SoftLabel erstellt und gespeichert")

    return train_soft_labels_reshaped

###########################################################################################################
#creating (or loading) teacher model
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('Model_1 training is starting...')
model = create_NN_with_temp(number_of_epochs = 50, show_model_architecture = True, show_grafics = False, show_train_eval = True, model_save_path = "./models/model_1_50epochs", temperature = 40, dataset = datasets.cifar10, class_names = names, with_softlabels = False)
print('Model_1 is done and saved!')

#model = models.load_model("./models/model_1_50epochs")

###########################################################################################################
#creating (or loading) Soft Labels
train_soft_labels = create_softlabels(datasets.cifar10, savepath = "softlabel_50epochs", model = model)

#train_soft_labels = np.load('softlabel_50epochs.npy')
#train_soft_labels = train_soft_labels.reshape(train_soft_labels.shape[0], -1)

##########################################################################################################
#creating (or loading) student model
print('Model_2 training is starting...')
model = create_NN_with_temp(number_of_epochs = 50, show_model_architecture = True, show_grafics = False, show_train_eval = True, model_save_path = "./models/model_2_50epochs", temperature = 40, dataset = datasets.cifar10, class_names = names, with_softlabels = True, softlabels = train_soft_labels)
print('Model_2 training is done!')

#model = models.load_model("./models/model_2_50epochs")

#########################################################################################################
#modifying the model to have temperature t=1 at test time
model.pop()
model.summary()

x = (model.layers[-1].output)
o = layers.Activation('softmax')(x)

model_d = models.Model(inputs = model.input, outputs=[o]) 
model_d.summary()
model_d.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=Adam(lr=0.001, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
model_d.save("./models/model_d_50epochs")