import librosa
import librosa.display as display
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from BeatLoader import BeatLoader
import Util as utils


class Dataset:
    def __init__(self):
        self.data_num = 10
        self.input_path = 'input/'
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        # Audio Sampling Rate: 22050 samples/sec
        self.sr = 0
        self.audios = []
        self.labels = []

        # features used
        self.mel_specs = []  # (128, 1292)
        self.mfccs = []  # (10, 1292)
        self.tempograms = []  #
        self.chromagrams = []  # (12, 1292)

        # helpers
        self.max_dim = -1
        self.hop_length = 512
        self.tempos = []

    def load_audios(self):
        for idx, genre in enumerate(self.genres):
            print(f'===== Load {genre.capitalize()} Starts =====')
            # for path in glob(f'{self.input_path}{genre}.*.wav'):
            path = f'{self.input_path}{genre}.00000.wav'
            y, sr = librosa.load(path=path, duration=30)
            self.sr = sr
            self.audios.append(y)
            self.labels.append(idx)

            self.calculate_features(y, path, genre)

            print(f'===== Load {genre.capitalize()} Ends =====')

        self.max_dim = utils.get_max_dim(self.mel_specs, self.mfccs, self.chromagrams)

    def calculate_features(self, y, path, genre):
        self.calculate_mel_spec(y)
        self.calculate_mfcc(y)
        self.calculate_tempogram(y)
        self.calculate_chromagram(y)

        # plot graph for first audio of each genre
        if utils.detail_mode(path):
            self.plot_mel_spec(len(self.audios) - 1, genre)
            self.plot_mfcc(len(self.audios) - 1, genre)
            self.plot_tempogram(len(self.audios) - 1, genre)
            self.plot_chromagram(len(self.audios) - 1, genre)

    # =================================================================
    #                           Calculations
    # =================================================================
    def calculate_mel_spec(self, y):
        mel_spec = librosa.feature.melspectrogram(y=y)
        self.mel_specs.append(mel_spec)

    def calculate_mfcc(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=10)
        self.mfccs.append(mfcc)

    def calculate_tempogram(self, y):
        oenv = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.sr, hop_length=self.hop_length)
        self.tempograms.append(tempogram)

        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=self.sr, hop_length=self.hop_length)[0]
        self.tempos.append(tempo)

    def calculate_chromagram(self, y):
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        self.chromagrams.append(chroma_cq)

    # =================================================================
    #                           Plotting
    # =================================================================
    def plot_graph(self, data, title, path, y_axis, x_axis, ax_format=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(title)
        img = display.specshow(data, x_axis=x_axis, y_axis=y_axis, sr=self.sr, hop_length=self.hop_length)
        fig.colorbar(img, ax=ax, format=ax_format)

        fig.savefig(path, bbox_inches='tight', pad_inches=0.25)
        fig.show()
        fig.clf()

    def plot_mel_spec(self, idx, genre):
        title = f'Mel Spectrogram Example for {genre.capitalize()} Genre'
        path = f'output/mel_spec/{genre}.png'
        data = librosa.power_to_db(np.abs(self.mel_specs[idx]), ref=np.max)

        self.plot_graph(data, title, path, x_axis='time', y_axis='mel', ax_format='%+2.0f dB')

    def plot_mfcc(self, idx, genre):
        title = f'MFCC Example for {genre.capitalize()} Genre'
        path = f'output/mfcc/{genre}.png'

        self.plot_graph(self.mfccs[idx], title, path, x_axis='time', y_axis='linear')

    def plot_tempogram(self, idx, genre):
        title = f'Tempogram Example for {genre.capitalize()} Genre'
        path = f'output/tempogram/{genre}.png'

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(title)
        librosa.display.specshow(self.tempograms[idx], sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='tempo', cmap='magma', ax=ax)
        ax.axhline(self.tempos[idx], color='w', linestyle='--', alpha=1, label=f'Estimated tempo={self.tempos[idx]}')
        ax.legend(loc='upper right')

        fig.savefig(path, bbox_inches='tight', pad_inches=0.25)
        fig.show()
        fig.clf()

    def plot_chromagram(self, idx, genre):
        title = f'Chromagram Example for {genre.capitalize()} Genre'
        path = f'output/chromagram/{genre}.png'

        self.plot_graph(self.chromagrams[idx], title, path, x_axis='time', y_axis='chroma')

    # =================================================================
    #                         Preprocess
    # =================================================================
    # mel_spec (12) + mfcc (10) + beat (5) + chromagram (12)
    def load_features(self, idx):
        idx_name = f'00{idx}' if idx <= 9 else f'0{idx}'
        if glob(f'output/features/{idx_name}.npy'):
            return np.array(utils.read_feat_from_files(idx_name))
        else:
            if len(self.audios) == 0:
                self.load_audios()

            # pad features to max
            mel_spec = utils.pad_data(self.mel_specs[idx], self.max_dim)
            mfcc = utils.pad_data(self.mfccs[idx], self.max_dim)
            rhythmic_feat = utils.pad_data(np.array(self.tempograms[idx]).reshape(-1, 1), self.max_dim)
            chromagram = utils.pad_data(self.chromagrams[idx], self.max_dim)

            feat = np.concatenate((mel_spec, mfcc, rhythmic_feat, chromagram), axis=0)

            if idx == 0:
                # --- Combining Features ---
                # > mel_spec: (128, 1292)
                # > mfccs: (10, 1292)
                # > rhythmic_feat: (5, 1292)
                # > chromagrams: (12, 1292)
                # > combined feat: (155, 1292)
                print('--- Combining Features ---')
                print(f'Padded all features to {self.max_dim}')
                print(f'> mel_spec: {mel_spec.shape}')
                print(f'> mfccs: {mfcc.shape}')
                print(f'> rhythmic_feat: {rhythmic_feat.shape}')
                print(f'> chromagrams: {chromagram.shape}')
                print(f'> combined feat: {feat.shape}')

            utils.write_feat_to_files(idx_name, feat)
            return np.array(feat)
