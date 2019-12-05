import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.layers import MaxPool2D, Reshape
from tensorflow.keras.layers import BatchNormalization, Permute
from tensorflow.keras.layers import UpSampling2D


# df = "channels_last"
# kernel = 3
# filter_size = 64
# pad = 1
# pool_size = 2

def Segnet(img_height, img_width, img_channels):

    inputs = Input((img_height, img_width, img_channels))

    # # encoder
    # model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    # model.add(Conv2D(filter_size, kernel, padding='valid', data_format=df))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))
    c1 = Conv2D(64, (3, 3), padding='same') (inputs)
    c1 = BatchNormalization() (c1)
    c1 = Activation('relu') (c1)
    p1 = MaxPool2D() (c1)

    # model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    # model.add(Conv2D(128, kernel, padding='valid', data_format=df))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))
    c2 = Conv2D(128, (3, 3), padding='same') (p1)
    c2 = BatchNormalization() (c2)
    c2 = Activation('relu') (c2)
    c2 = Conv2D(128, (3, 3), padding='same') (c2)
    c2 = BatchNormalization() (c2)
    c2 = Activation('relu') (c2)
    p2 = MaxPool2D() (c2)

    # model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    # model.add(Conv2D(256, kernel, padding='valid', data_format=df))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))
    c3 = Conv2D(256, (3, 3),  padding='same') (p2)
    c3 = BatchNormalization() (c3)
    c3 = Activation('relu') (c3)
    p3 = MaxPool2D() (c3)

    # model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    # model.add(Conv2D(512, kernel, padding='valid', data_format=df))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    c4 = Conv2D(512, (3, 3),padding='same') (p3)
    c4 = BatchNormalization() (c4)
    c4 = Activation('relu') (c4)

    # model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    # model.add(Conv2D(512, kernel, padding='valid', data_format=df))
    # model.add(BatchNormalization())
    c5 = Conv2D(512, (3, 3), padding='same') (c4)
    c5 = BatchNormalization() (c5)

    # model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))
    # model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    # model.add(Conv2D(256, kernel, padding='valid', data_format=df))
    # model.add(BatchNormalization())
    u1 = UpSampling2D(size=(2,2)) (c5)
    c6 = Conv2D(256, (3,3), padding='same') (u1)
    c6 = BatchNormalization() (c6)

    # model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))
    # model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    # model.add(Conv2D(128, kernel, padding='valid', data_format=df))
    # model.add(BatchNormalization())
    u2 = UpSampling2D(size=(2, 2)) (c6)
    c7 = Conv2D(128, (3,3), padding='same') (u2)
    c7 = BatchNormalization() (c7)

    # model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))
    # model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    # model.add(Conv2D(filter_size, kernel, padding='valid', data_format=df))
    # model.add(BatchNormalization())
    u3 = UpSampling2D(size=(2, 2)) (c7)
    c8 = Conv2D(64, (3,3), padding='same') (u3)
    c8 = BatchNormalization(scale=False) (c8)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

