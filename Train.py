from Dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


class Train:
    def __init__(self):
        self.dataset = Dataset()
        self.confusion_matrix = []

        self.X = [self.dataset.combine_features(idx) for idx in range(len(self.dataset.audios))]
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
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]