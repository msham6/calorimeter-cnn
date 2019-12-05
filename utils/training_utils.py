import tensorflow as tf
import numpy as np
import h5py
import pickle
import os

def add_noise_naive(images, noise=0.02):
    
    """
    Works for both 32x32 and 64x64 images
    """
    
    # image shapes --> (N runs, K layers=32, num_cells, num_cells, 1)
    num_runs = images.shape[0]
    num_layers = images.shape[1]
    num_cells = images.shape[2]

    new_images = images

    for run in range(num_runs):
        data_run = new_images[run, :, :, :, :]
        # shape --> (32, 64, 64, 1)
        for lay in range(2, num_layers):

            # Flatten data layer
            data_layer = data_run[lay, :, :, :]
            data_layer_flat = np.reshape(data_layer, (-1, 1))

            # Generate noise level mask
            mask = np.random.randint(low=0, high = num_cells*num_cells, size=int(noise*num_cells*num_cells))
            # Add noise to mask and reshape again
            data_layer_flat[mask] += 1.0
            data_layer_withnoise = np.reshape(data_layer_flat,(num_cells,num_cells))

            data_run[lay, :, :, 0] = data_layer_withnoise
        new_images[run, :, :, :, :] = data_run

    return new_images


def add_noise_naive_raw(images, noise=0.02):
    
    """
    Works for both 32x32 and 64x64 images
    """
    
    # image shapes --> (N runs, K layers=32, num_cells, num_cells, 1)
    num_runs = images.shape[0]
    num_layers = images.shape[1]
    num_cells = images.shape[2]

    new_images = images

    for run in range(num_runs):
        data_run = new_images[run, :, :, :, :]
        # shape --> (32, 64, 64, 1)
        for lay in range(num_layers):

            # Flatten data layer
            data_layer = data_run[lay, :, :, :]
            data_layer_flat = np.reshape(data_layer, (-1, 1))

            # Generate noise level mask
            mask = np.random.randint(low=0, high = num_cells*num_cells, size=int(noise*num_cells*num_cells))
            # Add noise to mask and reshape again
            data_layer_flat[mask] += 1.0
            data_layer_withnoise = np.reshape(data_layer_flat,(num_cells,num_cells))

            data_run[lay, :, :, 0] = data_layer_withnoise
        new_images[run, :, :, :, :] = data_run

    return new_images



def get_images_single_hit(direct, energy, num_runs, add_noise = True, noise=0.02):
    
    """
    
    Given a directory for the data, where the data is in the shape
    (N runs, K layers=30, num_cells, num_cells), this function
    reads the data, appends two entry point images to the start for
    every run, and then adds random noise to the images if required
    Output data shape = (N runs, K layers=32, 32, 32, 1)
    
    """

    file = '%.1fGeV_%iruns_data.h5' %(energy, num_runs)
    f = direct + file
    data_runs = h5py.File(f, 'r')['dataset_1'][:]
    
    # Define the entry label, just one in the center
    entry = np.zeros((32, 32))
    entry[16, 16] = 1
    new_data_runs = np.zeros((num_runs, 32, 32, 32))
    
    # Add entry label to images set for every run  
    for run in range(num_runs):

        # new data --> (N layers+2=32, num_cells, num_cells)
        # add entry label twice in images set
        new_data_by_run = np.zeros((32, 32, 32))
        new_data_by_run[0] = entry
        new_data_by_run[1] = entry

        # Data --> (N layers=30, 32, 32)
        # Assign input data to rest of the layers for every run
        data_by_run = data_runs[run, :, :, :]
        new_data_by_run[2:] = data_by_run
        new_data_runs[run] = new_data_by_run
        
    new_data_runs = np.expand_dims(new_data_runs, axis=-1)
    if add_noise is True:
        new_data_runs = add_noise_naive(new_data_runs, noise)
        
    return new_data_runs



def get_labels_single_hit(direct, energy, num_runs):
    
    """
    Same as above, excpets adds no noise. Makes all cluster 
    cells 1.0, others = 0.0
    Output data shape = (N runs, K layers=32, 32, 32, 1)
    """

    file = '%.1fGeV_%iruns_data.h5' %(energy, num_runs)
    f = direct + file
    data_runs = h5py.File(f, 'r')['dataset_1'][:]

    entry = np.zeros((32, 32))
    entry[16, 16] = 1
    new_data_runs = np.zeros((num_runs, 32, 32, 32))

    for run in range(num_runs):
        # Add entry point label twice 
        new_data_by_run = np.zeros((32, 32, 32))
        new_data_by_run[0] = entry
        new_data_by_run[1] = entry
        
        # All cluster cells > 0 = 1, else 0
        data_by_run = data_runs[run]
        data_by_run[data_by_run>0] = 1
        new_data_by_run[2:] = data_by_run
        new_data_runs[run] = new_data_by_run
    new_data_runs = np.expand_dims(new_data_runs, axis=-1)

    return new_data_runs


def get_images_single_hit_raw(direct, energy, num_runs, add_noise = True, noise=0.02):
    
    """
    
    Given a directory for the data, where the data is in the shape
    (N runs, K layers=30, num_cells, num_cells), this function
    reads the data, and then adds random noise to the images if required
    Output data shape = (N runs, K layers=30, 32, 32, 1)
    
    """

    file = '%.1fGeV_%iruns_data.h5' %(energy, num_runs)
    f = direct + file
    new_data_runs = h5py.File(f, 'r')['dataset_1'][:]
        
    new_data_runs = np.expand_dims(new_data_runs, axis=-1)
    if add_noise is True:
        new_data_runs = add_noise_naive_raw(new_data_runs, noise)
        
    return new_data_runs


def get_labels_single_hit_raw(direct, energy, num_runs):
    
    """
    Same as above, excpets adds no noise. Makes all cluster 
    cells 1.0, others = 0.0
    Output data shape = (N runs, K layers=30, 32, 32, 1)
    """

    file = '%.1fGeV_%iruns_data.h5' %(energy, num_runs)
    f = direct + file
    new_data_runs = h5py.File(f, 'r')['dataset_1'][:]
    new_data_runs[new_data_runs > 0] = 1
    new_data_runs = np.expand_dims(new_data_runs, axis=-1)

    return new_data_runs


def get_data_flat(direct, energies, num_runs, data_type, add_noise=True, noise=0.02):
    
    """
    Get the data as a sequence of images and not arranged by runs
    Output --> (num_runs*num_energies*num_layers, 32, 32, 1)
    """
    
    num_energies = len(energies)
    num_layers=32
    new_data_flat = np.zeros((1, 32, 32, 1))
    
    for energy in energies:
        
        if data_type == 'images':
            data_runs = get_images_single_hit(direct, energy, num_runs, add_noise = add_noise, noise=noise) 
        elif data_type == 'labels':
            data_runs = get_labels_single_hit(direct, energy, num_runs)
            
        data_flat = np.reshape(data_runs, (-1, 32, 32, 1))
        new_data_flat = np.concatenate((new_data_flat, data_flat))
    new_data_flat = new_data_flat[1:]
    return new_data_flat
        
    
def get_data_flat_raw(direct, energies, num_runs, data_type, add_noise=True, noise=0.02):
    
    """
    Get the data as a sequence of images and not arranged by runs
    Output --> (num_runs*num_energies*num_layers, 32, 32, 1)
    """
    
    num_energies = len(energies)
    num_layers=30
    new_data_flat = np.zeros((1, 32, 32, 1))
    
    for energy in energies:
        
        if data_type == 'images':
            data_runs = get_images_single_hit_raw(direct, energy, num_runs, add_noise = add_noise, noise=noise) 
        elif data_type == 'labels':
            data_runs = get_labels_single_hit_raw(direct, energy, num_runs)
            
        data_flat = np.reshape(data_runs, (-1, 32, 32, 1))
        new_data_flat = np.concatenate((new_data_flat, data_flat))
    new_data_flat = new_data_flat[1:]
    return new_data_flat




# ----------------------------- REDUNDANT ---------------------------------


# def train(in_model, train_dset, val_dset, num_epochs=1, is_training=False, print_every=100):

#     loss_fn = tf.keras.losses.BinaryCrossentropy()

#     model = tf.keras.models.clone_model(in_model)
#     optimizer = tf.keras.optimizers.Adam()

#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

#     val_loss = tf.keras.metrics.Mean(name='val_loss')
#     val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

#     t = 0
#     for epoch in range(num_epochs):

#         train_loss.reset_states()
#         train_accuracy.reset_states()

#         for x_np, y_np in train_dset:
#             with tf.GradientTape() as tape:

#                 # Use the model function to build the forward pass.
#                 scores = model(x_np, training=is_training)
#                 loss = loss_fn(y_np, scores)

#                 # Compute gradients and update parameters
#                 gradients = tape.gradient(loss, model.trainable_variables)
#                 optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#                 print(model.trainable_variables[0][0])

#                 # Update the metrics
#                 train_loss.update_state(loss)
#                 train_accuracy.update_state(y_np, scores)

#                 if t % print_every == 0:
#                     val_loss.reset_states()
#                     val_accuracy.reset_states()
#                     dice_coeffs = []
#                     for test_x, test_y in val_dset:
#                         # During validation at end of epoch, training set to False
#                         prediction = model.predict(test_x)
#                         t_loss = loss_fn(test_y, prediction)

#                         val_loss.update_state(t_loss)
#                         val_accuracy.update_state(test_y, prediction)

#                         a = tf.dtypes.cast(test_y, dtype='double')
#                         b = tf.dtypes.cast(prediction, dtype='double')
#                         dice = dice_coef(a, b)
#                         dice_coeffs.append(dice)

#                     dice_coeffs = np.asarray(dice_coeffs)
#                     template = 'Iteration {}, Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}, Dice coeffs:'
#                     print(template.format(t, epoch+1,
#                                          train_loss.result(),
#                                          train_accuracy.result()*100,
#                                          val_loss.result(),
#                                          val_accuracy.result()*100),
#                                          np.mean(dice_coeffs))
#                 t += 1
#     return model


# # data h5
# def get_data_h5(direct, num_runs, energies, kind):

#     data_files = []
#     # r=root, d=directories, f = files
#     for energy in energies:
#         filename = direct + '%.1fGeV_%iruns_' %(energy, num_runs)
#         filename += kind + '.h5'
#         data_files.append(filename)

#     data_images = []
#     for f in data_files:
# #         data = pickle.load(open(f, mode='rb'))
# #         data_images.append(data)
#         data = h5py.File(f, 'r')['dataset_1'][:]
#         data_images.append(data)

#     data_images = np.array(data_images)
#     num_cells = data_images.shape[3]; num_layers = data_images.shape[2]
#     data_images = data_images.reshape(-1, num_layers, num_cells, num_cells)
#     d = data_images.reshape(-1, num_cells, num_cells)
#     d = np.expand_dims(d, axis=-1)

#     if kind == 'labels':
#         d[d[:,:,:,0] > 0] = 1

#     return d

# # Data pickle
# def get_data_pickle(direct, num_runs, energies, kind):

#     data_files = []
#     # r=root, d=directories, f = files
#     for energy in energies:
#         filename = direct + '%i_%i_' %(energy, num_runs)
#         filename += kind
# #         if not os.path.exists(filename):
# #             continue
#         for r, d, files in os.walk(filename):
#             for file in files:
#                 data_files.append(os.path.join(r, file))

#     data_images = []
#     for f in data_files:
#         data = pickle.load(open(f, mode='rb'))
#         data_images.append(data)
#     data_images = np.array(data_images)

#     num_cells = data_images.shape[2]
#     d = data_images.reshape(-1, num_cells, num_cells)
#     d = np.expand_dims(d, axis=-1)

#     if kind == 'labels':
#         d[d[:,:,:,0] > 0] = 1

#     return d


# def arrange_by_run(inp_images, inp_labels, num_layers, num_cells):
    
#     """
    
#     Given images and labels in shape --> (N images, num_cells, num_cells, 1)
#     first inserts entry label and duplicates last layer, then returns
#     output --> (N runs, K layers, num_cells, num_cells, 1)

#     """

#     images = np.reshape(inp_images, (-1, num_layers, num_cells, num_cells, 1))
#     labels = np.reshape(inp_labels, (-1, num_layers, num_cells, num_cells, 1))

#     num_runs = images.shape[0]
#     new_images = np.zeros((num_runs, num_layers+2, num_cells, num_cells, 1))
#     new_labels = np.zeros((num_runs, num_layers+2, num_cells, num_cells, 1))

#     for run in range(num_runs):
#         run_images = images[run]
#         run_labels = labels[run]

#         # first entry label
#         entry_label = run_labels[0]
#         entry_label = np.expand_dims(entry_label, axis=0)

#         # last layer label
#         end_label = run_labels[29]
#         end_label = np.expand_dims(end_label, axis=0)

#         # last layer label
#         end_image = run_images[29]
#         end_image = np.expand_dims(end_image, axis=0)

#         # insert entry and last label to new run images
#         new_run_images = np.insert(run_images, 0, entry_label, axis=0)
#         new_run_images = np.insert(new_run_images, num_layers, end_image, axis=0)

#         # update labels
#         new_run_labels = np.insert(run_labels, 0, entry_label, axis=0)
#         new_run_labels = np.insert(new_run_labels, num_layers, end_label, axis=0)

#         # update total list
#         new_images[run] = new_run_images
#         new_labels[run] = new_run_labels

#     return new_images, new_labels