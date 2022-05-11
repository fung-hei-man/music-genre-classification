from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Dataset import Dataset
from utils import reshape_feats_to_1d
import numpy as np


class TrainWithSVC:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = np.array(self.dataset.load_features())
        self.Y = np.array(self.dataset.labels)

        self.name = 'SVM'
        self.model = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    def fit(self, x_train, y_train):
        print(f'fitting for {self.name}')
        data = reshape_feats_to_1d(x_train)
        self.model = self.model.fit(data, y_train)

        return self.model

    def predict(self, x_test):
        print(f'predicting with {self.name}')
        data = reshape_feats_to_1d(x_test)
        return self.model.predict(data)
