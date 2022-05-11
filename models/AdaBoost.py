from sklearn.ensemble import AdaBoostClassifier
from utils import reshape_feats_to_1d
import numpy as np


class TrainWithAdaBoost:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = np.array(self.dataset.load_features())
        self.Y = np.array(self.dataset.labels)

        self.name = 'Ada Boost'
        self.model = AdaBoostClassifier(n_estimators=100, random_state=0)

    def fit(self, x_train, y_train):
        print(f'fitting for {self.name}')
        data = reshape_feats_to_1d(x_train)
        self.model = self.model.fit(data, y_train)

        return self.model

    def predict(self, x_test):
        print(f'predicting with {self.name}')
        data = reshape_feats_to_1d(x_test)
        return self.model.predict(data)
