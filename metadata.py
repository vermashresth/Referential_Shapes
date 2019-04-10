import pickle
import numpy as np
import os

def does_shapes_onehot_metadata_exist(shapes_dataset):
    return (os.path.exists('shapes/{}/train.large.onehot_metadata.p'.format(shapes_dataset)) and
        os.path.exists('shapes/{}/val.onehot_metadata.p'.format(shapes_dataset)) and
        os.path.exists('shapes/{}/test.onehot_metadata.p'.format(shapes_dataset)))

def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out

def create_shapes_onehot_metadata(shapes_dataset):
    set_names = ['train.large', 'val', 'test']

    for set_name in set_names:
        meta = pickle.load(
            open('shapes/{}/{}.metadata.p'.format(shapes_dataset, set_name), 'rb')
        )

        compressed_images = np.zeros((len(meta), 5))

        for i, m in enumerate(meta):
            if type(m) is tuple:
                m = m[0] # Only grab the metadata of the first target (aka, the sender target)
            pos_h, pos_w = (np.array(m["shapes"]) != None).nonzero()
            pos_h, pos_w = pos_h[0], pos_w[0]
            color = m["colors"][pos_h][pos_w]
            shape = m["shapes"][pos_h][pos_w]
            size = m["sizes"][pos_h][pos_w]
            compressed_images[i] = np.array([color, shape, size, pos_h, pos_w])

        compressed_images = compressed_images.astype(np.int)

        one_hot_derivations = one_hot(compressed_images).reshape(
            compressed_images.shape[0], -1
        )

        pickle.dump(one_hot_derivations, open('shapes/{}/{}.onehot_metadata.p'.format(shapes_dataset, set_name), 'wb'))

def load_shapes_onehot_metadata(shapes_dataset):
    train_metadata = pickle.load(open('shapes/{}/train.large.onehot_metadata.p'.format(shapes_dataset), 'rb'))
    val_metadata = pickle.load(open('shapes/{}/val.onehot_metadata.p'.format(shapes_dataset), 'rb'))
    test_metadata = pickle.load(open('shapes/{}/test.onehot_metadata.p'.format(shapes_dataset), 'rb'))

    return train_metadata, val_metadata, test_metadata