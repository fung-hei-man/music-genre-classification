from Train import Train
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    # dataset = Dataset()
    train = Train()

    gmm = GaussianMixture(n_components=10)
    train.train_with_model(gmm)
