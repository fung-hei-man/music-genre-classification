from sklearn.mixture import GaussianMixture

from Dataset import Dataset
from utils import reshape_feats_to_1d
import numpy as np


class TrainWithGMM:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = np.array(self.dataset.load_features())
        self.Y = np.array(self.dataset.labels)

        self.name = 'GMM'
        self.model = GaussianMixture(n_components=10, covariance_type='diag', means_init=self.calculate_init_mean())

    def fit(self, x_train, _):
        print(f'fitting for {self.name}')
        data = reshape_feats_to_1d(x_train)
        self.model = self.model.fit(data)

        return self.model

    def predict(self, x_test):
        print(f'predicting with {self.name}')
        data = reshape_feats_to_1d(x_test)
        return self.model.predict(data)

    def calculate_init_mean(self):
        flattened_x = reshape_feats_to_1d(self.X)
        means = [flattened_x[self.Y == genre].mean(axis=0) for genre in self.dataset.genres]
        print(f'init_mean: {means}')
        return means
