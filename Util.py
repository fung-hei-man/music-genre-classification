import numpy as np


def detail_mode(path):
    return path.endswith('classical.00000.wav') or path.endswith('rock.00000.wav')


def write_feat_to_files(idx_name, combined_feats):
    # np.save(f'output/features/{idx_name}.npy', combined_feats)
    pass


def read_feat_from_files(idx_name):
    feats = np.load(f'output/features/{idx_name}.npy')
    return feats


def get_max_dim(mel_specs, mfccs, chromagrams):
    max1 = max([i.shape[1] for i in mel_specs])
    max2 = max([i.shape[1] for i in mfccs])
    max3 = max([i.shape[1] for i in chromagrams])

    return max(max1, max2, max3)


def pad_data(data, max_dim):
    if data.ndim == 1:
        return np.pad(data, (0, max_dim - len(data)))

    elif data.ndim == 2:
        return np.array([np.pad(item, (0, max_dim - len(item))) for item in data])