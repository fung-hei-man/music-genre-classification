from Dataset import Dataset
from Train import Train
from models.AdaBoost import TrainWithAdaBoost
from models.GMM import TrainWithGMM
from models.KNeighbours import TrainWithKNeighbors
from models.NeuralNetwork import TrainWithNeuralNetwork
from models.SVC import TrainWithSVC
import time


if __name__ == '__main__':
    dataset = Dataset()
    train = Train(dataset)

    print("=== GMM ===")
    start_time = time.time()
    gmm = TrainWithGMM(dataset)
    train.train_with_model(gmm)
    print(f'>>> Training time: {time.time() - start_time}')

    print('=== KNeighbors ===')
    start_time = time.time()
    knn = TrainWithKNeighbors(dataset)
    train.train_with_model(knn)
    print(f'>>> Training time: {time.time() - start_time}')

    print('=== Neural Network ===')
    start_time = time.time()
    nn = TrainWithNeuralNetwork(dataset)
    train.train_with_model(nn)
    print(f'>>> Training time: {time.time() - start_time}')

    print('=== SVM ===')
    start_time = time.time()
    svm_model = TrainWithSVC(dataset)
    train.train_with_model(svm_model)
    print(f'>>> Training time: {time.time() - start_time}')

    print('=== AdaBoost ===')
    start_time = time.time()
    adaBoost = TrainWithAdaBoost(dataset)
    train.train_with_model(adaBoost)
    print(f'>>> Training time: {time.time() - start_time}')