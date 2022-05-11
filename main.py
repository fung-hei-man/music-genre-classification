from GMMTrain import TrainWithGMM
from Train import Train

if __name__ == '__main__':
    # dataset = Dataset()
    train = Train()

    print("=== GMM ===")
    gmm_model = TrainWithGMM()
    train.train_with_model(gmm_model)
