"""
Implementation of UNet architecture.

For a quick overview of UNet check this post
https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
This implementation is based on the code presented there.

"""

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Convolution2DTranspose, Cropping2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping


def create_model():
    def convolution_block(input_tensor, n_filters, kernel_size=3):
        x = Convolution2D(filters=n_filters, kernel_size=kernel_size,
                          activation='relu', padding='valid')(input_tensor)
        x = Convolution2D(filters=n_filters, kernel_size=kernel_size,
                          activation='relu', padding='valid')(x)
        return x

    n_filters = 64

    inputs = Input(shape=(572, 572, 1))

    conv1 = convolution_block(inputs, n_filters * 1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = convolution_block(pool1, n_filters * 2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = convolution_block(pool2, n_filters * 4)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = convolution_block(pool3, n_filters * 8)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = convolution_block(pool4, n_filters * 16)

    up6 = Convolution2DTranspose(filters=n_filters * 8, kernel_size=3, strides=2,
                                 activation='relu', padding='same')(conv5)
    crop6 = Cropping2D(cropping=((4, 4), (4, 4)))(conv4)
    concat6 = concatenate([crop6, up6], axis=3)
    conv6 = convolution_block(concat6, n_filters * 8)

    up7 = Convolution2DTranspose(filters=n_filters * 4, kernel_size=3, strides=2,
                                 activation='relu', padding='same')(conv6)
    crop7 = Cropping2D(cropping=((16, 16), (16, 16)))(conv3)
    concat7 = concatenate([crop7, up7], axis=3)
    conv7 = convolution_block(concat7, n_filters * 4)

    up8 = Convolution2DTranspose(filters=n_filters * 2, kernel_size=3, strides=2,
                                 activation='relu', padding='same')(conv7)
    crop8 = Cropping2D(cropping=((40, 40), (40, 40)))(conv2)
    concat8 = concatenate([crop8, up8], axis=3)
    conv8 = convolution_block(concat8, n_filters * 2)

    up9 = Convolution2DTranspose(filters=n_filters * 1, kernel_size=3, strides=2,
                                 activation='relu', padding='same')(conv8)
    crop9 = Cropping2D(cropping=((88, 88), (88, 88)))(conv1)
    concat9 = concatenate([crop9, up9], axis=3)
    conv9 = convolution_block(concat9, n_filters * 1)

    outputs = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # print(model.summary())
    return model


class UNet(object):
    def __init__(self):
        self.model = create_model()

    def save_weights(self, filename='model.hd5'):
        self.model.save_weights(filename)

    def load_weights(self, filename='model.hd5'):
        self.model.load_weights(filename)

    def plot_model(self, show_shapes=True, show_layer_names=True, filename='model.png'):
        from keras.utils import plot_model
        plot_model(self.model,
                   show_shapes=show_shapes,
                   show_layer_names=show_layer_names,
                   to_file=filename)

    def fit(self, x, y, **kwargs):
        """fits the model to `(x, y)`

        :param x: shape `(n_samples, n_features)`
        :param y: shape `(n_samples,)` or `(n_samples, n_outputs)
        :param kwargs: legal arguments are the arguments of `Model.fit`
        :return: history: details about the training history at each epoch
        """
        x = x.values.astype('float32')
        x = x.reshape(x.shape[0], 572, 572, 1)

        if 'validation_data' in kwargs:
            vx = kwargs['validation_data'][0].values.astype('float32')
            vx = vx.reshape(vx.shape[0], 572, 572, 1)
            vy = kwargs['validation_data'][1]
            kwargs['validation_data'] = (vx, vy)

        # tensorboard
        tensorboard = TensorBoard(log_dir='./logs',
                                  histogram_freq=0,
                                  batch_size=32,
                                  update_freq='epoch')
        # learning rate reduction
        annealer = ReduceLROnPlateau(monitor='val_acc',
                                     patience=3,
                                     verbose=1,
                                     factor=0.5,
                                     min_lr=0.00001)
        # earlystopping
        earlystopping = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=0,
                                      verbose=0,
                                      mode='auto',
                                      baseline=None,
                                      restore_best_weights=False)
        kwargs['callbacks'] = [tensorboard, annealer, earlystopping]

        return self.model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        pass

    def score(self, x, y, **kwargs):
        pass


def main():
    unet = UNet()
    unet.plot_model()


if __name__ == '__main__':
    main()
