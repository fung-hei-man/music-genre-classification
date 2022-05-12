from sklearn.neighbors import KNeighborsClassifier

from utils import reshape_feats_to_1d
import numpy as np


class TrainWithKNeighbors:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = np.array(self.dataset.load_features())
        self.Y = np.array(self.dataset.labels)

        self.name = 'KNN'
        self.model = KNeighborsClassifier(n_neighbors=10)

    def fit(self, x_train, y_train):
        print(f'fitting for {self.name}')
        data = reshape_feats_to_1d(x_train)
        self.model = self.model.fit(data, y_train)

        return self.model

    def predict(self, x_test):
        print(f'predicting with {self.name}')
        data = reshape_feats_to_1d(x_test)
        return self.model.predict(data)
