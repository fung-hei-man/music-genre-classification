from Dataset import Dataset
from Train import Train
from models.KNeighbours import TrainWithKNeighbors
from models.SVC import TrainWithSVC

if __name__ == '__main__':
    dataset = Dataset()
    train = Train(dataset)

    # print("=== GMM ===")
    # gmm_model = TrainWithGMM()
    # train.train_with_model(gmm_model)

    # print('=== KNeighbors ===')
    # knn_model = TrainWithKNeighbors(dataset)
    # train.train_with_model(knn_model)

    print('=== SVM ===')
    svm_model = TrainWithSVC(dataset)
    train.train_with_model(svm_model)
