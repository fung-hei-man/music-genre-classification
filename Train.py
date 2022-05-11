from Dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np


class Train:
    def __init__(self):
        self.dataset = Dataset()
        self.confusion_matrix = []

        self.X = np.array(self.dataset.load_features())
        self.Y = np.array(self.dataset.labels)
        self.label_encoder = self.preprocess_labels()

    def preprocess_labels(self):
        label_encoder = LabelEncoder()
        label_encoder.fit(self.Y)
        # print(label_encoder.classes_)

        return label_encoder

    def train_with_model(self, model):
        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(self.X):
            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]

            # transform from text label to numeric labels
            y_train = self.label_encoder.transform(y_train)
            y_test = self.label_encoder.transform(y_test)

            _, counts = np.unique(y_train)
            self.confusion_matrix.append(confusion_matrix(y_test, y_text_pred))

        accuracy = np.mean(np.array(self.confusion_matrix).diagonal())
        print(f'>>> accuracy: {accuracy}')