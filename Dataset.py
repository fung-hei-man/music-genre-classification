import librosa
import librosa.display as display
import glob
import numpy as np
import matplotlib.pyplot as plt

from BeatLoader import BeatLoader


def detail_mode(path):
    return path.endswith('classical.00000.wav') or path.endswith('rock.00000.wav')


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
        self.rhythmic_feats = []  # 5-dim, without sum
        self.pitch_feats = []  # 5-dim

        self.load_audios()

    def load_audios(self):
        for idx, genre in enumerate(self.categories):
            print(f'===== Load {genre.capitalize()} Starts =====')
            path = f'{self.input_path}{genre}.00000.wav'
            # for path in glob.glob(f'{self.input_path}{genre}.*.wav'):
            y, sr = librosa.load(path=path, duration=30)
            self.sr = sr
            self.audios.append(y)
            self.labels.append(idx)

            self.calculate_features(y, path, genre)

            print(f'===== Load {genre.capitalize()} Ends =====')

    def calculate_features(self, y, path, genre):
        self.calculate_stft(y)
        self.calculate_mfcc(y)
        beat_loader = self.calculate_rhythmic_feat(path)

        # plot graph for first audio of each genre
        if detail_mode(path):
            self.plot_stft(len(self.audios) - 1, genre)
            self.plot_mfcc(len(self.audios) - 1, genre)
            self.plot_beats(beat_loader, genre)

    # =================================================================
    #                           Calculations
    # =================================================================
    def calculate_stft(self, y):
        stft = librosa.stft(y=y)
        self.stft.append(stft)

    def calculate_mfcc(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=10)
        self.mfcc.append(mfcc)

    def calculate_rhythmic_feat(self, path):
        beat_loader = BeatLoader(path, detail_mode(path))
        beat_loader.load_beats()
        beat_loader.calculate_beat_histogram()

        rhythmic_feats = beat_loader.get_rhythmic_feats()
        self.rhythmic_feats.append(rhythmic_feats)

        return beat_loader

    # =================================================================
    #                           Plotting
    # =================================================================
    def plot_graph(self, data, title, path, y_axis, x_axis):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(title)
        img = display.specshow(data, x_axis=x_axis, y_axis=y_axis, sr=self.sr)
        fig.colorbar(img, ax=ax)

        fig.savefig(path, bbox_inches='tight', pad_inches=0.25)
        fig.show()
        fig.clf()

    def plot_stft(self, idx, genre):
        title = f'STFT Example for {genre.capitalize()} Genre'
        path = f'output/stft/{genre}.png'
        data = librosa.amplitude_to_db(np.abs(self.stft[idx]), ref=np.max)

        self.plot_graph(data, title, path, x_axis='time', y_axis='linear')

    def plot_mfcc(self, idx, genre):
        title = f'MFCC Example for {genre.capitalize()} Genre'
        path = f'output/mfcc/{genre}.png'

        self.plot_graph(self.mfcc[idx], title, path, x_axis='time', y_axis='linear')

    def plot_beats(self, beat_loader, genre):
        beat_loader.plot_wave_and_beat(title=f'Beat Estimation Example for {genre.capitalize()}', path=f'output/beat/{genre}.png')
        beat_loader.plot_beat_hist(title=f'Beat Histogram Example for {genre.capitalize()}', path=f'output/beat_hist/{genre}.png')

