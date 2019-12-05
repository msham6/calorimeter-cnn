#!/usr/bin/env python
# coding: utf-8

# In[7]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import concatenate

dataformat = 'channels_last'
def d3UNet(img_depth, img_height, img_width, img_channels):

    inputs = Input((img_depth, img_height, img_width, img_channels))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='elu', kernel_initializer='he_normal',
                padding='same', data_format = dataformat, name='Conv1_1') (s)
    c1 = Dropout(0.2) (c1)
    c1 = Conv3D(16, kernel_size=(3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv1_2') (c1)
    p1 = MaxPool3D(pool_size=(2, 2, 2), name='Pool1') (c1)

    #Conv2
    c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv2_1') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv2_2') (c2)
    p2 = MaxPool3D(pool_size=(2, 2, 2), name='Pool2') (c2)

    # Conv3
    c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv3_1') (p2)
    c3 = Dropout(0.3) (c3)
    c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv3_2') (c3)
    p3 = MaxPool3D((2, 2, 2), name='Pool3') (c3)

    # Conv4
    c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv4_1') (p3)
    c4 = Dropout(0.3) (c4)
    c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv4_2') (c4)
    p4 = MaxPool3D(pool_size=(2, 2, 2), name='Pool4') (c4)

    # Conv5
    c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv5_1') (p4)
    c5 = Dropout(0.4) (c5)
    c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv5_2') (c5)

    # Up6
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='ConvT1', data_format=dataformat) (c5)
    u6 = concatenate([u6, c4], axis=4)
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv6_1') (u6)
    c6 = Dropout(0.3) (c6)
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv6_2') (c6)

    # Up7
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='ConvT2', data_format=dataformat) (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv7_1') (u7)
    c7 = Dropout(0.3) (c7)
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv7_2') (c7)

    # Up8
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='ConvT3', data_format=dataformat) (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv8_1') (u8)
    c8 = Dropout(0.2) (c8)
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv8_2') (c8)

    # Up9
    u9 = Conv3DTranspose(filters=16, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='ConvT4', data_format=dataformat) (c8)
    u9 = concatenate([u9, c1], axis=4)
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv9_1') (u9)
    c9 = Dropout(0.2) (c9)
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same',
                data_format = dataformat, name = 'Conv9_2') (c9)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid', name='Output') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model





