from sklearn.ensemble import RandomForestClassifier

from utils import reshape_feats_to_1d
import numpy as np


class TrainWithRandomForest:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = np.array(self.dataset.load_features())
        self.Y = np.array(self.dataset.labels)

        self.name = 'Random Forest'
        self.model = RandomForestClassifier(random_state=0)

    def fit(self, x_train, y_train):
        print(f'fitting for {self.name}')
        data = reshape_feats_to_1d(x_train)
        self.model = self.model.fit(data, y_train)

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
