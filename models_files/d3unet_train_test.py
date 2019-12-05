from d3unet_import3 import d3UNet
import numpy as np
import h5py
from training import dice_coef, dice_coef_loss, get_data_h5, get_data_pickle, set_by_run
import tensorflow as tf


en1 = np.array([0.1])
en2 = np.arange(5, 45.0, 5.0)
train_energies = np.append(en1, en2)

train_images = get_data_pickle('simulations_pickle\\', 100, train_energies, 'data')
train_labels = get_data_pickle('simulations_pickle\\', 100, train_energies, 'labels')
print(train_images.shape)
print(train_labels.shape)

val_energies = [13, 18]
val_images = get_data_pickle('simulations_pickle\\', 10, val_energies, 'data')
val_labels = get_data_pickle('simulations_pickle\\', 10, val_energies, 'labels')
print(val_images.shape)
print(val_labels.shape)


train_images_run, train_labels_run = set_by_run(train_images, train_labels, 30, 32)
val_images_run, val_labels_run = set_by_run(val_images, val_labels, 30, 32)

print(train_images_run.shape)
print(train_labels_run.shape)


img_depth, img_width, img_height, img_channels = 32, 64, 64, 1

model = d3UNet(img_depth, img_width, img_height, img_channels)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='binary_crossentropy', 
              metrics = [dice_coef])

# checkpoint
filepath = "saved_models\\unet-3d\\unet3d_trial-ep{epoch:02d}-dice{val_dice_coef:.2f}.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', min_delta=0.0, patience=3, verbose=1, mode='max')
callbacks_list = [checkpoint, earlystop]

history = model.fit(multiple_train_images, multiple_train_labels, 
          batch_size=1, 
          epochs=1,
          validation_data = (multiple_val_images[val_mask], multiple_val_labels[val_mask]),
          shuffle=True,
          callbacks=callbacks_list
         )