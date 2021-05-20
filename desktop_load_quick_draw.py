import urllib.request
import numpy as np
import tensorflow as tf
import os

# Items, die geladen werden sollen
items = [
    'apple',
    'asparagus',
    'banana',
    'blackberry',
    'blueberry', 
    'broccoli',
    'carrot',
    'grapes',
    'peanut',
    'pear',
    'pineapple',
    'strawberry',
    'watermelon'
]

# Pfad, in dem die Ergebnisdateien gespeichert werden
DATA_PATH = './data/'


def download():
    # Laden der Datei auf die lokale Platte
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    download_path = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for i in items:
        path = download_path + i + '.npy'
        print(path)
        urllib.request.urlretrieve(path, DATA_PATH+i+'.npy')


def check_data_downloaded():
    data_found = True
    for i in items:
        path = DATA_PATH + i + '.npy'
        if not os.path.exists(path):
            data_found = False
    if not data_found:
        download()


def save_data(max_elements_per_item=1000, test_data_in_percent=0.1):
    check_data_downloaded()
    # Aus den einzelnen Dateien einen Datensatz zum Testen und Validieren erzeugen
    x = np.empty([0, 784], dtype=np.float32)
    y = np.empty([0], dtype=np.int32)

    for i, item in enumerate(items):
        data = np.load(DATA_PATH+item+'.npy')
        data = data[0:max_elements_per_item, :]
        labels = np.full(data.shape[0], i)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)
    
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    test_size = int(x.shape[0]*test_data_in_percent)

    x_test = x[0:test_size, :]
    x_test = x_test.reshape(x_test.shape[0], 28, 28)
    y_test = y[0:test_size]

    x_train = x[test_size:, :]
    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    y_train = y[test_size:]

    np.save(DATA_PATH+'x_train.npy', x_train)
    np.save(DATA_PATH+'y_train.npy', y_train)
    np.save(DATA_PATH+'x_test.npy', x_test)
    np.save(DATA_PATH+'y_test.npy', y_test)


def load_data():
    return np.load(DATA_PATH+'x_train.npy'), np.load(DATA_PATH+'y_train.npy')


def load_test_data():
    return np.load(DATA_PATH+'x_test.npy'), np.load(DATA_PATH+'y_test.npy')


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    save_data(4000, 0.1)
