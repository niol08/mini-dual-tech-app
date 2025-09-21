from __future__ import division, print_function
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, add, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization, Lambda
from keras import backend as K
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable(package="custom")
def zeropad(x):
    """ 
    zeropad and zeropad_output_shapes are from 
    https://github.com/awni/ecg/blob/master/ecg/network.py
    """
    y = tf.zeros_like(x)
    return tf.concat([x, y], axis=2)

@register_keras_serializable(package="custom")
def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[2] *= 2
    return tuple(shape)


def ECG_model(config):
    """ 
    implementation of the model in https://www.nature.com/articles/s41591-018-0268-3 
    also have reference to codes at 
    https://github.com/awni/ecg/blob/master/ecg/network.py 
    and 
    https://github.com/fernandoandreotti/cinc-challenge2017/blob/master/deeplearn-approach/train_model.py
    """
    def first_conv_block(inputs, config):
        layer = Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        shortcut = MaxPooling1D(pool_size=1,
                      strides=1)(layer)

        layer =  Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(config.drop_rate)(layer)
        layer =  Conv1D(filters=config.filter_length,
                        kernel_size=config.kernel_size,
                        padding='same',
                        strides=1,
                        kernel_initializer='he_normal')(layer)
        return add([shortcut, layer])

    def main_loop_blocks(layer, config):
        filter_length = config.filter_length
        n_blocks = 15
        for block_index in range(n_blocks):

            subsample_length = 2 if block_index % 2 == 0 else 1
            shortcut = MaxPooling1D(pool_size=subsample_length)(layer)

            if block_index % 4 == 0 and block_index > 0 :
                shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
                filter_length *= 2

            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer =  Conv1D(filters= filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides=subsample_length,
                            kernel_initializer='he_normal')(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(config.drop_rate)(layer)
            layer =  Conv1D(filters= filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides= 1,
                            kernel_initializer='he_normal')(layer)
            layer = add([shortcut, layer])
        return layer

    def output_block(layer, config):
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Flatten()(layer)
        outputs = Dense(len_classes, activation='softmax')(layer)
        model = Model(inputs=inputs, outputs=outputs)
        
        adam = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
        model.compile(optimizer= adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.summary()
        return model

    classes = ['N','V','/','A','F','~']
    len_classes = len(classes)

    inputs = Input(shape=(config.input_size, 1), name='input')
    layer = first_conv_block(inputs, config)
    layer = main_loop_blocks(layer, config)
    return output_block(layer, config)
