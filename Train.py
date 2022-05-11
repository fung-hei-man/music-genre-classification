from Dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


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

        return label_encoder

    def train_with_model(self, model):
        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(self.X):
            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]

            # transform from text label to numeric labels
            y_train = self.label_encoder.transform(y_train)
            y_test = self.label_encoder.transform(y_test)

            model.fit(x_train)
            y_text_pred = model.predict(x_test)

            _, counts = np.unique(y_test, return_counts=True)
            counts = counts.reshape(len(self.dataset.genres), 1)
            self.confusion_matrix.append(confusion_matrix(y_test, y_text_pred) / counts)

        cm_mean = np.array(self.confusion_matrix).mean(axis=0)
        accuracy = cm_mean.diagonal().mean() * 100
        print(f'>>> accuracy: {accuracy}')

        self.plot_confusion_matrix(cm_mean, model.name)

    def plot_confusion_matrix(self, cm, model_name):
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.dataset.genres)
        disp.plot()

        plt.title(f'Confusion Matrix for {model_name}')
        plt.xticks(rotation=75)

        plt.savefig(f'output/cm/{model_name}.png', bbox_inches='tight', pad_inches=0.3)
        plt.show()
        plt.clf()
