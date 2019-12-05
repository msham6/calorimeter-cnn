#!/usr/bin/env python
# coding: utf-8

# In[7]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import MaxPool3D, BatchNormalization, Activation
from tensorflow.keras.layers import concatenate

dataformat = 'channels_last'
def d3UNet(img_depth, img_height, img_width, img_channels, num_classes):

    inputs = Input((img_depth, img_height, img_width, img_channels))
    # s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                padding='same', data_format = dataformat, name='Conv1_1') (inputs)
    c1 = BatchNormalization() (c1)
    c1 = Activation(activation='relu') (c1)
    c1 = Dropout(0.2) (c1)
    c1 = Conv3D(64, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv1_2') (c1)
    c1 = BatchNormalization() (c1)
    c1 = Activation(activation='relu') (c1)
    p1 = MaxPool3D(pool_size=(2, 2, 2), name='Pool1') (c1)

    #Conv2
    c2 = Conv3D(64, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv2_1') (p1)
    c2 = BatchNormalization() (c2)
    c2 = Activation(activation='relu') (c2)
    c2 = Dropout(0.1) (c2)
    c2 = Conv3D(128, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv2_2') (c2)
    c2 = BatchNormalization() (c2)
    c2 = Activation(activation='relu') (c2)
    p2 = MaxPool3D(pool_size=(2, 2, 2), name='Pool2') (c2)

    # Conv3
    c3 = Conv3D(128, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv3_1') (p2)
    c3 = BatchNormalization() (c3)
    c3 = Activation(activation='relu') (c3)
    c3 = Dropout(0.3) (c3)
    c3 = Conv3D(256, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv3_2') (c3)
    c3 = BatchNormalization() (c3)
    c3 = Activation(activation='relu') (c3)
    p3 = MaxPool3D((2, 2, 2), name='Pool3') (c3)

    # Conv4
    c4 = Conv3D(256, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv4_1') (p3)
    c4 = BatchNormalization() (c4)
    c4 = Activation(activation='relu') (c4)
    c4 = Dropout(0.3) (c4)
    c4 = Conv3D(512, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv4_2') (c4)
    c4 = BatchNormalization() (c4)
    c4 = Activation(activation='relu') (c4)

    # Up5
    u5 = Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same', name='ConvT1', data_format=dataformat) (c4)
    u5 = concatenate([u5, c3], axis=4)
    c5 = Conv3D(256, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv5_1') (u5)
    c5 = BatchNormalization() (c5)
    c5 = Activation(activation='relu') (c5)
    c5 = Dropout(0.3) (c5)
    c5 = Conv3D(256, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv5_2') (c5)
    c5 = BatchNormalization() (c5)
    c5 = Activation(activation='relu') (c5)

    # Up6
    u6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='ConvT2', data_format=dataformat) (c5)
    u6 = concatenate([u6, c2])
    c6 = Conv3D(128, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv6_1') (u6)
    c6 = BatchNormalization() (c6)
    c6 = Activation(activation='relu') (c6)
    c6 = Dropout(0.3) (c6)
    c6 = Conv3D(128, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv6_2') (c6)
    c6 = BatchNormalization() (c6)
    c6 = Activation(activation='relu') (c6)

    # Up7
    u7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='ConvT3', data_format=dataformat) (c6)
    u7 = concatenate([u7, c1])
    c7 = Conv3D(64, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv7_1') (u7)
    c7 = BatchNormalization() (c7)
    c7 = Activation(activation='relu') (c7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv3D(64, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv7_2') (c7)
    c7 = BatchNormalization() (c7)
    c7 = Activation(activation='relu') (c7)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid', name='Output') (c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model





