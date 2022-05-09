import librosa
import librosa.display as display
import glob
import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self):
        self.input_path = 'input/'
        self.categories = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        # Audio Sampling Rate: 22050 samples/sec
        self.sr = 0
        self.audios = []
        self.labels = []

        # features used
        self.stft = []  # 12-dim
        self.mfcc = []  # 10-dim
        self.rhythmicFeat = []  # 6-dim
        self.pitchFeat = []  # 5-dim

        self.load_audios()

    def load_audios(self):
        for idx, genre in enumerate(self.categories):
            path = f'{self.input_path}{genre}.00000.wav'
            # for path in glob.glob(f'{self.input_path}{genre}.*.wav'):
            y, sr = librosa.load(path=path, duration=30)
            self.sr = sr
            self.audios.append(y)
            self.labels.append(idx)
            print(f'Loaded all {genre} audios')

            # calculate features
            self.calculate_stft(y)
            self.calculate_mfcc(y)

            # plot graph for first audio of each genre
            if path.endswith(".00000.wav"):
                self.plot_stft(len(self.audios) - 1, genre)
                self.plot_mfcc(len(self.audios) - 1, genre)

    # =================================================================
    #                           Calculations
    # =================================================================
    def calculate_stft(self, y):
        stft = librosa.stft(y=y)
        print(stft.shape)
        self.stft.append(stft)

    def calculate_mfcc(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=10)
        print(mfcc.shape)
        self.mfcc.append(librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=10))

    # =================================================================
    #                           Plotting
    # =================================================================
    def plot_graph(self, data, title, path, y_axis, x_axis):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(title)
        img = display.specshow(data, x_axis=x_axis, y_axis=y_axis, sr=self.sr)
        fig.colorbar(img, ax=ax)

        fig.show()
        fig.savefig(path, bbox_inches='tight', pad_inches=0.25)
        fig.clf()

    def plot_stft(self, idx, genre):
        title = f'STFT Sample for {genre.capitalize()} Genre'
        path = f'output/stft/{genre}.png'
        data = librosa.amplitude_to_db(np.abs(self.stft[idx]), ref=np.max)

        self.plot_graph(data, title, path, x_axis='time', y_axis='linear')

    def plot_mfcc(self, idx, genre):
        title = f'MFCC Sample for {genre.capitalize()} Genre'
        path = f'output/mfcc/{genre}.png'
        data = librosa.amplitude_to_db(np.abs(self.stft[idx]), ref=np.max)

        self.plot_graph(self.mfcc[idx], title, path, x_axis='time', y_axis='linear')
