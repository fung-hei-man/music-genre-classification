from Dataset import Dataset
from Train import Train
from models.GMM import TrainWithGMM
from models.RandomForest import TrainWithRandomForest
from models.KNeighbours import TrainWithKNeighbors
from models.NeuralNetwork import TrainWithNeuralNetwork
from models.SVC import TrainWithSVC
import time


if __name__ == '__main__':
    dataset = Dataset()
    train = Train(dataset)

    print('=== GMM ===')
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

    print("=== Random Forest ===")
    start_time = time.time()
    rf = TrainWithRandomForest(dataset)
    train.train_with_model(rf)
    print(f'>>> Training time: {time.time() - start_time}')

    print('=== SVC ===')
    start_time = time.time()
    svc = TrainWithSVC(dataset)
    train.train_with_model(svc)
    print(f'>>> Training time: {time.time() - start_time}')
