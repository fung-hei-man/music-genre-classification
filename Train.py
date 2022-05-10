from Dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np


class Train:
    def __init__(self):
        self.dataset = Dataset()
        self.confusion_matrix = []

        self.X = [self.dataset.load_features(idx) for idx in range(self.dataset.data_num)]
        self.Y = self.dataset.labels
        self.Y_encoder = self.preprocess_labels()

    def preprocess_labels(self):
        label_encoder = LabelEncoder()
        label_encoder.fit(self.Y)
        # print(label_encoder.classes_)

        return label_encoder

    def train_with_model(self, model):
        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(self.X):
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]

            trained_model = model.fit(x_train)
            y_text_pred = trained_model.predict(x_test)

            _, counts = np.unique(y_train)
            self.confusion_matrix.append(confusion_matrix(y_test, y_text_pred))

        accuracy = np.mean(np.array(self.confusion_matrix).diagonal())
        print(f'>>> accuracy: {accuracy}')