from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from scipy.io import loadmat
import numpy as np
import h5py

import os


def get_nyu_dir():
    key = 'NYU_PATH'
    if key in os.environ:
        dataset_dir = os.environ[key]
        if not os.path.isdir(dataset_dir):
            raise Exception('%s directory does not exist' % key)
        return dataset_dir
    else:
        raise Exception('%s environment variable not set.' % key)


def load_data(data_path):
    return h5py.File(data_path, 'r')


def load_split(split_path):
    from scipy.io import loadmat
    return loadmat(split_path)


def get_data_path(version='v1'):
    base = get_nyu_dir()
    if version == 'v1':
        return os.path.join(base, 'v1', 'nyu_depth_data_labeled.mat')
    elif version == 'v2':
        return os.path.join(base, 'v2', 'nyu_depth_v2_labeled.mat')
    else:
        raise ValueError('version "%s" unrecognized' % version)


def get_split_path(version='v1'):
    base = get_nyu_dir()
    if version == 'v1':
        return os.path.join(base, 'v1', 'splits_classification.mat')
    elif version == 'v2':
        return os.path.join(base, 'v2', 'splits.mat')
    else:
        raise ValueError('version "%s" unrecognized' % version)


def get_names(f, indices=None):
    """
    Gets weirdly saved string values in names. Go figure.
    https://stackoverflow.com/questions/28541847/how-convert-this-type-of-data-hdf5-object-reference-to-something-more-readable

    Args:
        f: hdf5 file
        indices: if None, returns all entries, otherwise just the specified
            ones.

    Returns:
        list of strings, saved weirdly in names, or names[indices] if indices
            is provided.
    """  # NOQA
    ns = np.squeeze(f['names'])
    if indices is None:
        return [''.join(chr(i) for i in f[ni]) for ni in ns]
    elif hasattr(indices, '__iter__'):
        return [''.join(chr(i) for i in f[ns[i]]) for i in indices]
    else:
        try:
            return ''.join(chr(i) for i in f[ns[indices]])
        except Exception:
            raise ValueError('Invalid indices "%s"' % str(indices))


class NyuLoader(object):
    def __init__(self, version='v2'):
        self._version = version
        self._data = None
        self._split = None
        self._label_names = None

    @property
    def version(self):
        return self._version

    @property
    def data_path(self):
        return get_data_path(self._version)

    @property
    def split_path(self):
        return get_split_path(self._version)

    @property
    def label_names(self):
        if self._label_names is None:
            names = get_names(self._data)
            names.insert(0, 'unlabelled')
            self._label_names = tuple(names)
        return self._label_names

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    @property
    def example_dimensions(self):
        s = self._data['images'].shape
        return (s[3], s[2])

    def open(self):
        if self._data is None:
            self._data = load_data(self.data_path)
            self._split = load_split(self.split_path)
        else:
            raise RuntimeError('Loader already open.')

    def close(self):
        if self._data is None:
            raise RuntimeError('Loader already closed.')
        else:
            self._data.close()
            self._data = None
            self._split = None

    def split(self, mode):
        if mode == 'train':
            key = 'trainNdxs'
        elif mode in ('test', 'predict', 'eval'):
            key = 'testNdxs'
        else:
            raise ValueError('Unsupported mode "%s"' % mode)
        return self._split[key][:, 0]

    def get_example(self, index):
        return NyuExample(self, index)

    def get_image(self, index):
        return np.array(self._data['images'][index]).transpose((2, 1, 0))

    def get_depth(self, index):
        return np.array(self._data['depths'][index]).T

    def get_raw_depths(self, index):
        return np.array(self._data['rawDepths'][index]).T

    def get_labels(self, index):
        return np.array(self._data['labels'][index]).T

    def __len__(self):
        return self._data['images'].shape[0]


class NyuExample(object):
    def __init__(self, loader, index):
        self._loader = loader
        self._index = index

    @property
    def image(self):
        return self._loader.get_image(self._index)

    @property
    def depth(self):
        return self._loader.get_depth(self._index)

    @property
    def raw_depth(self):
        return self._loader.get_raw_depth(self._index)

    @property
    def labels(self):
        return self._loader.get_labels(self._index)


def label_mapper(original_labels, super_categories):
    misc = len(super_categories)
    original_indices = {label: i for i, label in enumerate(original_labels)}
    # print([n for n in original_labels if 'roof' in n])
    # exit()
    map_indices = misc*np.ones(len(original_labels))
    for i, cats in enumerate(super_categories):
        for k in cats:
            map_indices[original_indices[k]] = i

    def call(labelled_image, dtype=None):
        return map_indices[labelled_image]

    return call
