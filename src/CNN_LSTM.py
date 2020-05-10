from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Activation, AveragePooling2D

from keras.regularizers import l2
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.layers import LSTM
import numpy as np
import glob,os
# from keras import applications
# from cv2 import imread, resize
import matplotlib.pyplot as plt

#after feature extraction is done, directly call test() to do all the training and testing.
batch_size = 30

TRAIN_TYPE, VALIDATE_TYPE, TEST_TYPE = 'train', 'validation', 'test'

BASE_PATH = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_PATH, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_OPT_DIR = os.path.join(DATA_DIR, 'trainOF')
VALIDATION_OPT_DIR = os.path.join(DATA_DIR, 'validationOF')
TEST_OPT_DIR = os.path.join(DATA_DIR, 'testOF')

labels = os.listdir(TRAIN_DIR)

def load_conv_net(weights='imagenet', include_top=False):
    base_model = InceptionV3(weights=weights, include_top=include_top, input_shape=(224,224,3))
    x = base_model.output
    average_pool = GlobalAveragePooling2D()(x)
    classifier = Dense(1000, kernel_regularizer=l2(0.01))(average_pool)
    extra_1 = Dense(4096, activation='relu')(classifier)

    if include_top:
        predictions = Activation('softmax', name='predictions')(extra_1)
        model = Model(inputs=base_model.input, outputs=predictions)
    else:
        model = Model(inputs=base_model.input, outputs=average_pool)
    return model

def count_total_images(dir_path):
    labels_dir = os.listdir(dir_path)
    count = 0
    for label in labels_dir:
        train_vid_dir = os.listdir(os.path.join(dir_path, label))
        for train_vid in train_vid_dir:
            if train_vid[-3:] != 'npy':
                frames = os.listdir(os.path.join(dir_path, label, train_vid))
                if len(frames) < 30:
                    print(train_vid)
                count += len(frames)
    return count

def save_features(model, d_dir, labels, set_type: str = '', optical=False):
    """
    extract features using CNN model, and save as .npy file
    :param model: pre-trained CNN model
    :param d_dir: data directory
    :param labels: video labels
    :param set_type: TRAIN_TYPE, VALIDATION_TYPE or TEST_TYPE
    :param optical:
    :return:
    """
    datagen = ImageDataGenerator(rescale=1. / 255)
    d_generator = datagen.flow_from_directory(
            d_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            classes=labels)

    x_generator, y_label = None, None
    batch = 0
    d_img_num = count_total_images(d_dir)

    for x, y in d_generator:
        if batch == int(d_img_num/batch_size):
            break
        print("total batchs:", int(d_img_num/batch_size), "predict on batch:", batch)
        batch += 1
        if x_generator is None:
            x_generator = model.predict_on_batch(x)
            y_label = y
            print(y)
        else:
            x_generator = np.append(x_generator,model.predict_on_batch(x),axis=0)
            y_label = np.append(y_label, y, axis=0)

    if optical:
        x_name = f'video_x_{set_type}_optical.npy'
        y_name = f'video_y_{set_type}_optical.npy'
        # np.save(open(f'video_x_{set_type}_optical.npy', 'wb'), x_generator)
        # np.save(open(f'video_y_{set_type}_optical.npy','wb'), y_label)
    else:
        x_name = f'video_x_{set_type}.npy'
        y_name = f'video_y_{set_type}.npy'
    print("save_features.data type:", set_type)
    np.save(open(x_name, 'wb'), x_generator)
    np.save(open(y_name, 'wb'), y_label)

def save_extract_features(base_model, labels, optical_flow=False):
    """
    extract train/validation/test frame features using CNN models,
    and save as .npy file
    :param base_model: model for feature extraction
    :param labels: video labels
    :param optical_flow:
    :return: nothing
    """
    save_features(base_model, TRAIN_DIR, labels, set_type=TRAIN_TYPE, optical=optical_flow)
    save_features(base_model, VALIDATION_DIR, labels, set_type=VALIDATE_TYPE, optical=optical_flow)
    save_features(base_model, TEST_DIR, labels, set_type=TEST_TYPE, optical=optical_flow)

def load_features(data_type='train', optical_flow=False):
    # vid_data, vid_labels = None, None
    if optical_flow:
        vid_data = np.load(open(f'video_x_{data_type}_optical.npy', 'rb'))
        vid_labels = np.load(open(f'video_y_{data_type}_optical.npy', 'rb'))
    else:
        vid_data = np.load(f'video_x_{data_type}.npy')
        vid_labels = np.load(f'video_y_{data_type}.npy')
    return vid_data, vid_labels

def load_data(optical_flow=False):
    train_data, train_labels = load_features(TRAIN_TYPE, optical_flow)
    validation_data, validation_labels = load_features(VALIDATE_TYPE, optical_flow)

    train_data = train_data.reshape(train_data.shape[0]//30, 30, train_data.shape[1])
    validation_data = validation_data.reshape(validation_data.shape[0]//30, 30, validation_data.shape[1])

    t_labels, val_labels = [], []
    for i, label in enumerate(train_labels):
        if i % 30 == 0:
            t_labels.append(label)

    train_labels = np.array(t_labels)
    for i, label in enumerate(validation_labels):
        if i % 30 == 0:
            val_labels.append(label)
    validation_labels = np.array(val_labels)

    train_data, train_labels = shuffle(train_data, train_labels)
    validation_data, validation_labels = shuffle(validation_data, validation_labels)

    print(f"final train_data shape:{train_data.shape}")
    print(f"final train_labels shape:{train_labels.shape}")

    return train_data, train_labels, validation_data, validation_labels


def LSTM_layers(train_data):
    """used fully connected layers, SGD optimizer and
    checkpoint to store the best weights"""
    model = Sequential()
    model.add(LSTM(512, dropout=0.2, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.00001*train_data.shape[1], decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def train_model(epochs=500, weights_file: str = 'model.h5'):
    train_data, train_labels, validation_data,  validation_labels = load_data()
    model = LSTM_layers(train_data)
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                 ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True, verbose=0) ]
    history = model.fit(train_data,train_labels,
                        validation_data=(validation_data,validation_labels),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        shuffle=True,
                        verbose=1)
    model.load_weights(weights_file)
    return history, model

# def train_with_optical(train_dir, validation_dir, labels, epochs=500, timedistributed=False, optical=True):
# 	if optical:
# 		train_data,train_labels,validation_data,validation_labels = concatenate_two_stream()
# 	else:
# 		train_data,train_labels,validation_data,validation_labels = load_data(optical)
# 	model = LSTM_layers_with_optical(train_data,train_labels,validation_data,validation_labels, timedistributed=timedistributed)
# 	callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
# 	nb_epoch = epochs
# 	history = model.fit(train_data,train_labels,
# 						validation_data=(validation_data,validation_labels),
# 						batch_size=batch_size,
# 						nb_epoch=nb_epoch,
# 						callbacks=callbacks,
# 						shuffle=True,
# 						verbose=1)
# 	return history, model

def preprocess_data():
    pass

def test_vids(optical=False, weights_file: str = 'model.h5', optical_flow=False):
    test_data, test_labels = load_features(TEST_TYPE, optical_flow)
    print(test_data.shape)
    test_data = test_data.reshape(test_data.shape[0]//30, 30, test_data.shape[1])
    t_labels = []
    for i, label in enumerate(test_labels):
        if i % 30 == 0:
            t_labels.append(label)
    test_labels = np.array(t_labels)
    test_data,test_labels = shuffle(test_data,test_labels)
    model = None
    if optical:
        # TODO: optical training function
        pass
    else:
        history, model = train_model(weights_file=weights)
    results = model.evaluate(test_data, test_labels)
    print("Test Accuracy:", results[1])

def test(model, weights_file: str = 'model.h5', optical_flow=False):
    t_labels = []
    test_data, test_labels = load_features(TEST_TYPE, optical_flow)
    #print(test_data.shape)
    test_data = test_data.reshape(test_data.shape[0]//30, 30, test_data.shape[1])
    for i, label in enumerate(test_labels):
        if i % 30 == 0:
            t_labels.append(label)
    test_labels = np.array(t_labels)
    test_data, test_labels = shuffle(test_data, test_labels)
    model.load_weights(weights_file)
    results = model.evaluate(test_data, test_labels)
    print("Test Accuracy:", results[1])

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('Training and validation accuracy.png')

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('Training and validation loss.png')



if __name__ == '__main__':
    cnn = load_conv_net('imagenet')
    save_extract_features(cnn, labels)
    weights = 'CNN_5LSTMs.h5'
    print(weights)
    history, model = train_model(epochs=100, weights_file=weights)
    plot_history(history)
    test(model, weights_file=weights)