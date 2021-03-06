from sklearn.neural_network import MLPClassifier

from utils import reshape_feats_to_1d
import numpy as np


class TrainWithNeuralNetwork:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = np.array(self.dataset.load_features())
        self.Y = np.array(self.dataset.labels)

        self.name = 'Neural Network'
        self.model = MLPClassifier(hidden_layer_sizes=(150, 100, 100, 100, 50), random_state=1,
                                   max_iter=1000, early_stopping=True, n_iter_no_change=20,
                                   learning_rate_init=0.001)

    def fit(self, x_train, y_train):
        print(f'fitting for {self.name}')
        data = reshape_feats_to_1d(x_train)
        self.model = self.model.fit(data, y_train)

        return self.model

    def predict(self, x_test):
        print(f'predicting with {self.name}')
        data = reshape_feats_to_1d(x_test)
        return self.model.predict(data)
