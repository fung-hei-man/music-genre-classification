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
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        # Audio Sampling Rate: 22050 samples/sec
        self.sr = 0
        self.audios = []
        self.labels = []

        # features used
        self.mel_specs = []  # (128, 1292)
        self.mfccs = []  # (10, 1292)
        self.rhythmic_feats = []  # (5, 1) -> (5, 1292)
        self.chromagrams = []  # (12, 1292)

        self.load_audios()

    def load_audios(self):
        for idx, genre in enumerate(self.genres):
            # genre = 'classical'
            # idx = 0
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
        self.calculate_mel_spec(y)
        self.calculate_mfcc(y)
        beat_loader = self.calculate_rhythmic_feat(path)
        self.calculate_chromagram(y)

        # plot graph for first audio of each genre
        if detail_mode(path):
            self.plot_mel_spec(len(self.audios) - 1, genre)
            self.plot_mfcc(len(self.audios) - 1, genre)
            self.plot_beats(beat_loader, genre)
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

    def calculate_rhythmic_feat(self, path):
        beat_loader = BeatLoader(path, detail_mode(path))
        beat_loader.load_beats()
        beat_loader.calculate_beat_histogram()

        rhythmic_feats = beat_loader.get_rhythmic_feats()
        self.rhythmic_feats.append(rhythmic_feats)

        return beat_loader

    def calculate_chromagram(self, y):
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        self.chromagrams.append(chroma_cq)

    # =================================================================
    #                           Plotting
    # =================================================================
    def plot_graph(self, data, title, path, y_axis, x_axis, ax_format=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(title)
        img = display.specshow(data, x_axis=x_axis, y_axis=y_axis, sr=self.sr)
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

    def plot_beats(self, beat_loader, genre):
        beat_loader.plot_wave_and_beat(title=f'Beat Estimation Example for {genre.capitalize()}',
                                       path=f'output/beat/{genre}.png')
        beat_loader.plot_beat_hist(title=f'Beat Histogram Example for {genre.capitalize()}',
                                   path=f'output/beat_hist/{genre}.png')

    def plot_chromagram(self, idx, genre):
        title = f'Chromagram Example for {genre.capitalize()} Genre'
        path = f'output/chromagram/{genre}.png'

        self.plot_graph(self.chromagrams[idx], title, path, x_axis='time', y_axis='chroma')

    # =================================================================
    #                         Preprocess
    # =================================================================
    # mel_spec (12) + mfcc (10) + beat (5) + chromagram (12)
    def combine_features(self, idx):
        rhythmic_feats = np.array(self.rhythmic_feats[idx]).reshape(-1, 1)
        rhythmic_feats = np.array([np.pad(feat, (0, 1291)) for feat in rhythmic_feats])
        feat = np.concatenate((self.mel_specs[idx], self.mfccs[idx], rhythmic_feats, self.chromagrams[idx]), axis=0)

        if idx == 0:
            # --- Combining Features ---
            # > mel_spec: (128, 1292)
            # > mfccs: (10, 1292)
            # > rhythmic_feats: (5, 1292)
            # > chromagrams: (12, 1292)
            # > combined feat: (155, 1292)
            print('--- Combining Features ---')
            print('> mel_spec: ', end='')
            # print(self.mel_spec[idx])
            print(self.mel_specs[idx].shape)
            print('> mfccs: ', end='')
            # print(self.mfccs[idx])
            print(self.mfccs[idx].shape)
            print('> rhythmic_feats: ', end='')
            # print(self.rhythmic_feats[idx])
            print(rhythmic_feats.shape)
            print('> chromagrams: ', end='')
            # print(self.chromagrams[idx])
            print(self.chromagrams[idx].shape)
            print(f'> combined feat: {feat.shape}')

        return feat
