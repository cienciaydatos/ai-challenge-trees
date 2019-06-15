import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Cropping2D
from tensorflow.keras.layers import MaxPooling2D, Dropout, UpSampling2D, Input, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from utils import *

class UNET:
    def __init__(self, config):
        self.config = config
        self.metadata = pd.read_csv(self.config['model']['metadata_path'])
        self.model = self.build_model()

    def build_model(self):
        def crop_shape(down, up):
            ch = int(down[1] - up[1])
            cw = int(down[2] - up[2])
            ch1, ch2 = ch // 2, int(math.ceil(ch / 2))
            cw1, cw2 = cw // 2, int(math.ceil(cw / 2))

            return (ch1, ch2), (cw1, cw2)

        def get_shape(x):
            return tuple(x.get_shape().as_list())

        def conv2d_block(inputs, filters=16):
            c = inputs
            for _ in range(2):
                c = Conv2D(filters, (3,3), activation='relu', padding='valid') (c)
            return c

        input = Input((572, 572, 3))
        x = input
        # Downsampling path
        down_layers = []
        filters = 64
        for _ in range(4):
            x = conv2d_block(x, filters)
            down_layers.append(x)
            x = MaxPooling2D((2, 2), strides=2) (x)
            filters *= 2 # Number of filters doubled with each layer

        x = conv2d_block(x, filters)

        for conv in reversed(down_layers):
            filters //= 2
            x = Conv2DTranspose(filters, (2, 2), strides=(2, 2),
                                padding='same') (x)


            ch, cw = crop_shape(get_shape(conv), get_shape(x))
            conv = Cropping2D((ch, cw)) (conv)

            x = concatenate([x, conv])
            x = conv2d_block(x, filters)

        output = Conv2D(2, (1, 1), activation='softmax') (x)

        return Model(input, output)

    def preprocess(self, img, mask=None, padding_mode='CONSTANT'):
        insize = self.config['model']['input_size']
        ousize = self.config['model']['output_size']
        num_classes = len(self.config['model']['labels'])
        img = tf.image.resize(img, (ousize, ousize))
        img /= 255.0
        mask = tf.image.resize(mask, (ousize, ousize))
        mask /= 255.0

        pad = [[(insize - ousize) // 2] * 2] * 2 + [[0, 0]]
        img = tf.pad(img, pad, padding_mode)
        return img, mask

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def save_weights(self, weight_path):
        self.model.save_weights(weight_path)

    def _create_data(self, test=False):
        BATCH_SIZE = self.config['train']['batch_size']
        NUM_EPOCHS = self.config['train']['num_epochs']
        metadata = self.metadata

        train_source = build_source_from_metadata(
            metadata,
            self.config['model']['data_path'],
            'train'
        )

        train_data = make_dataset(
            train_source,
            mask = np.load(self.config['model']['mask_path']),
            training=True,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            num_parallel_calls=8,
            preprocess=lambda x, y: self.preprocess(x, y)
        )

        if test:
            test_source = build_source_from_metadata(
                self.metadata,
                self.config['model']['data_path'],
                'test'
            )
            test_data = make_dataset(
                test_source,
                mask = np.load(self.config['model']['mask_path']),
                training=False,
                batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                num_parallel_calls=-1,
                preprocess=lambda x, y: self.preprocess(x, y)
            )

            return train_data, test_data

        val_source = build_source_from_metadata(
            metadata,
            self.config['model']['data_path'],
            'val'
        )

        val_data = make_dataset(
            val_source,
            mask = np.load(self.config['model']['mask_path']),
            training=False,
            batch_size=4,
            num_epochs=NUM_EPOCHS,
            num_parallel_calls=8,
            preprocess=lambda x, y: self.preprocess(x, y)
        )

        return train_data, val_data

    def jaccard_loss(self, y_true, y_pred, smooth=100):
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
        sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

    def train(self):
        LR = self.config['train']['learning_rate']
        BATCH_SIZE = self.config['train']['batch_size']
        NUM_EPOCHS = self.config['train']['num_epochs']
        METRICS = self.config['train']['metrics']
        _CALLBACKS = self.config['train']['callbacks']

        self.model.compile(loss=self.jaccard_loss,
                           optimizer=optimizers.Adam(LR),
                           metrics=METRICS)
        train_data, val_data = self._create_data()

        CALLBACKS = [] if not _CALLBACKS \
            else [EarlyStopping(patience=10),
                  CSVLogger('log.csv'),
                  TerminateOnNaN(),
                  ReduceLROnPlateau(),
                  ModelCheckpoint('chpts/w.{epoch:02d}.h5')]

        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=NUM_EPOCHS,
            steps_per_epoch=BATCH_SIZE,
            validation_steps=4,
            callbacks=CALLBACKS)

    def evaluate(self):
        train_data, test_data = self._create_data(test=True)

        print('train')
        self.model.evaluate(train_data)
        print('test')
        self.model.evaluate(test_data)

    def predict(self, img):
        return self.model.predict(img)
