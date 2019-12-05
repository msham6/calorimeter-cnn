import tensorflow as tf
import numpy as np


smooth = 1
def dice_coef(y_true, y_pred):
    """
    Dice coefficient given true and predicted values
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


# Metric function
def dice_coef_np(y_true, y_pred):
    y_true = np.reshape(y_true, (-1, 1)) 
    y_pred =  np.reshape(y_pred, (-1, 1)) 
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def iou_coef_np(y_pred, y_true):
    y_true = np.reshape(y_true, (-1, 1)) 
    y_pred =  np.reshape(y_pred, (-1, 1)) 
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true + y_pred) - intersection
    iou_score = intersection/ union
    return iou_score


def get_coef_by_layer(images, labels, model, coef_type, num_layers=30):
    coef_dict = {}
    coefs = []
    for layer in range(num_layers):
        coef_dict['layer'+str(layer)] = []

    num_images = images.shape[0]
    for ind in range(num_images):
        layer = np.mod(ind, num_layers)

        # Get one image, label and prediction --> (1, 32, 32, 1)
        image = images[ind: ind+1, :, :, :]
        label = labels[ind: ind+1, :, :, :]
        pred = model.predict(image)
        
        if coef_type == 'dice':
            coef = dice_coef_np(label, pred)
        elif coef_type == 'iou':
            coef = iou_coef_np(label, pred)
        coefs.append(coef)
        coef_dict['layer'+str(layer)].append(coef)
    return coefs, coef_dict