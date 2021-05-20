# worker.py
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

WORKERS = ['raspi1', 'raspi2']
START_PORT = 54321
DATA_PATH = './data/'
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

print('TensorFlow Version {}'.format(tf.__version__))


def get_worker_list_with_ports():
    worker_list = []
    for index, hostname in enumerate(WORKERS):
        worker_list.append("{}:{:d}".format(hostname, START_PORT+index))
    return worker_list


def set_environment():
    hostname = os.uname().nodename
    print(hostname)
    tf_config = {
        'cluster': {
            'worker': get_worker_list_with_ports()
        },
        'task': {'type': 'worker', 'index': WORKERS.index(hostname)}
    }
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    print(os.environ['TF_CONFIG'])


def load_data():
    return np.load(DATA_PATH+'x_train.npy').astype(np.float32), np.load(DATA_PATH+'y_train.npy').astype(np.int32)


def load_test_data():
    return np.load(DATA_PATH+'x_test.npy').astype(np.float32), np.load(DATA_PATH+'y_test.npy').astype(np.int32)


def prepare_dataset(batch_size):
    x_train, y_train = load_data()

    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)

    return train_dataset


def prepare_test_dataset(batch_size):
    x_test, y_test = load_test_data()
    x_test = x_test.astype(np.float32) / np.float32(255)
    y_test = y_test.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size=batch_size)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    return dataset


def build_and_compile_model(number_of_items=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(number_of_items, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def load_model(test_dataset):
    new_model = tf.keras.models.load_model('model.h5')
    predictions = new_model.predict(test_dataset.unbatch().batch(3).take(1))
    print('predictions shape:', predictions.shape)


def train(dataset, validation_dataset):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = build_and_compile_model(len(items))
        model.summary()
        model.fit(dataset, epochs=32, steps_per_epoch=256, validation_data=validation_dataset)
        # Model speichern
        model.save('model.h5')  # HDF5


def evaluate(test_dataset):
    model = build_and_compile_model(len(items))
    test_loss, test_acc = model.evaluate(test_dataset, batch_size=64)
    print('test loss: {}, test acc: {}'.format(test_loss, test_acc))


NUM_WORKERS = len(WORKERS)
PER_WORKER_BATCH_SIZE = 8  # 64
GLOBAL_BATCH_SIZE = PER_WORKER_BATCH_SIZE * NUM_WORKERS

if os.uname().nodename in WORKERS:
    set_environment()

if __name__ == "__main__":
    train_dataset = prepare_dataset(GLOBAL_BATCH_SIZE)
    test_dataset = prepare_test_dataset(PER_WORKER_BATCH_SIZE)
    train(train_dataset, test_dataset)
    evaluate(test_dataset)
    load_model(test_dataset)
