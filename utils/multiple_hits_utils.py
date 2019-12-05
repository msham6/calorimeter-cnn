import numpy as np
from utils.training_utils import get_images_single_hit, get_labels_single_hit, add_noise_naive
import pickle
import h5py


def save_multiple_hits(direct, name, images, labels, run_dict):

    imgs_directory = direct + name + "_images.h5"
    f = h5py.File(imgs_directory, "w")
    f.create_dataset('dataset_1', dtype='f', data=images)
    f.close()
    print("* Data saved! *")

    labels_directory = direct + name + "_labels.h5"
    f2 = h5py.File(labels_directory, "w")
    f2.create_dataset('dataset_1', dtype='f', data=labels)
    f2.close()
    print("* Labels saved! *")

    dict_direct = direct + name + "_dict.p"
    with open(dict_direct, 'wb') as handle:
        pickle.dump(run_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('* Dictionary saved! *')


def read_multiple_hits(direct, name, num_runs=500, img_size=48, add_noise=False, noise=0.02, read_dict=False, predict=False, read_images=True):
    
    images = None
    if read_images is True:
        imgs_directory = direct + name + "_images.h5"
        images = h5py.File(imgs_directory, 'r')['dataset_1'][:]
        if add_noise is True:
            images = add_noise_naive(images, noise)
        if img_size == 48:
            images = images[:, :, 8:56, 8:56, :]
    
    labels = None
    if predict is False:
        labels_directory = direct + name + "_labels.h5"
        labels = h5py.File(labels_directory, 'r')['dataset_1'][:]
        if img_size == 48:
            labels = labels[:, :, 8:56, 8:56, :]
    
    my_dict = None
    if read_dict == True:
        dict_direct = direct + name + "_dict.p"
        my_dict = pickle.load(open(dict_direct, mode='rb'))
        my_dict = fix_dict(my_dict, num_runs)
    
    return images, labels, my_dict


def fix_dict(runs_dict, num_runs):
    
    new_runs_dict = {}
    for run in range(num_runs):
        d = runs_dict[str(run)]
        d['center_differences'] = np.sqrt(np.square(d['particle 2 center'][0]) + np.square(d['particle 2 center'][1]))
        new_runs_dict[str(run)] = d
    return new_runs_dict
    