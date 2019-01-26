import pickle

num_classes = 11
image_shape = [224,224,3]

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.misc import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Conv2D
from keras.models import Model
from keras.optimizers import Adam

class Metrics(Callback):

    def __init__(self,df=None, validation_generator=None):
        super()
        self.y_true = validation_df['one_hot_label'].values
        self.y_true = np.array([np.array(xi) for xi in self.y_true])
        self.validation_generator = validation_generator
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        print("Train begin")

    def on_epoch_end(self, epoch, logs={}):
        val_predict = self.model.predict_generator(self.validation_generator,steps=3430)
        val_targ = self.y_true
        _val_f1 = f1_score(val_targ.argmax(axis=1), val_predict.argmax(axis=1), average='macro')
        _val_recall = recall_score(val_targ.argmax(axis=1), val_predict.argmax(axis=1), average='micro')
        _val_precision = precision_score(val_targ.argmax(axis=1), val_predict.argmax(axis=1), average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("Epoch Endd")
        print(' — val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        
        return

def show_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    image = cv2.imread(addr)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    
    #image = imread(addr)
    #image = resize(image, (image_shape[0], image_shape[1]), mode='reflect')
    return image

def get_label(image_path,df):
    return df.loc[image_path]['one_hot_label']

def image_generator(df, batch_size = 8):
    
    while True:

        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a=df.index.values, size=batch_size)
        batch_images = []
        batch_labels = [] 
          
        # Read in each input, perform preprocessing and get labels

        for input_path in batch_paths:
            
            image = load_image(input_path)
            label = get_label(input_path,df)
            
            batch_images.append(image)
            batch_labels.append(label)

        # Return a tuple of (input,output) to feed the network

        batch_x = np.array(batch_images)
        batch_y = np.array(batch_labels)
        
        yield(batch_x, batch_y)
        
def create_model():

    pretrain_model = MobileNet(input_shape=image_shape, weights='imagenet', include_top=False)    

    input_tensor = Input(shape=image_shape)
    bn = BatchNormalization()(input_tensor)

    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(input_tensor, output)

def __main__():

    # Pandas dataframe containing image locations and labels
    with open('df.pickle','rb') as f:
        df = pickle.load(f)

    with open('validation_df.pickle','rb') as f:
        validation_df = pickle.load(f)

    labels = dict(zip(range(11), ['Bread', 
              'Dairy product', 
              'Dessert', 
              'Egg', 
              'Fried food', 
              'Meat', 
              'Noodles/Pasta', 
              'Rice', 
              'Seafood', 'Soup', 
              'Vegetable/Fruit']))

    print(labels)
    
    validation_datagen = ImageDataGenerator()

    validation_generator = validation_datagen.flow_from_directory(
        directory="data/val/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False
    )        
    
    
    model = create_model()    

    model.layers[2].trainable = False
    model.compile(optimizer=Adam(1e-3),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    model.summary()

    metrics = Metrics(validation_df,validation_generator)
    history = model.fit_generator(image_generator(df,batch_size=10),
                              steps_per_epoch=1000,
                              epochs=5,
                              callbacks=[metrics]
                             )
    #show_history(history)
    model.save_weights('mobilenet_baseline_n_5.model')
    
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        directory="data/eval/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )

    prediction = model.predict_generator(test_generator,steps=3430,
                                         verbose=1
                                        )
    