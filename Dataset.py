import librosa
import librosa.display as display
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import utils


class Dataset:
    def __init__(self):
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.sr = 0
        self.audios = []
        self.features = []
        self.labels = []

        # features used
        self.mel_specs = []  # (128, beat#)
        self.mfccs = []  # (10, beat#)
        self.tempograms = []  # (384, beat#)
        self.chromagrams = []  # (12, beat#)

        # helpers
        self.input_path = 'input/'
        self.output_graph_path = 'output/graph'
        self.max_dim = -1
        self.hop_length = 512
        self.tempos = []

    def load_audios(self):
        for idx, genre in enumerate(self.genres):
            for path in glob(f'{self.input_path}{genre}.*.wav'):
                # path = f'{self.input_path}{genre}.00000.wav'
                y, sr = librosa.load(path=path, duration=30)
                self.sr = sr
                self.audios.append(y)

                self.calculate_features(y, path, genre)

            print(f'===== Load {genre.capitalize()} Complete =====')
        self.max_dim = utils.get_max_dim(self.mel_specs, self.mfccs, self.chromagrams)

    def calculate_features(self, y, path, genre):
        beat_frames = self.get_beats(y)
        mel_spec = self.calculate_mel_spec(y, beat_frames)
        mfcc = self.calculate_mfcc(y, beat_frames)
        tempogram = self.calculate_tempogram(y, beat_frames)
        chromagram = self.calculate_chromagram(y, beat_frames)

        # plot graph for first audio of each genre
        if utils.detail_mode(path):
            self.plot_mel_spec(mel_spec, genre)
            self.plot_mfcc(mfcc, genre)
            self.plot_tempogram(tempogram, len(self.audios) - 1, genre)
            self.plot_chromagram(chromagram, genre)

    # =================================================================
    #                           Calculations
    # =================================================================
    def get_beats(self, y):
        _, beat_frames = librosa.beat.beat_track(y=y, sr=self.sr)
        return beat_frames

    def calculate_mel_spec(self, y, beat_frames):
        mel_spec = librosa.feature.melspectrogram(y=y)
        beat_mel_spec = librosa.util.sync(mel_spec, beat_frames)

        self.mel_specs.append(beat_mel_spec)
        return mel_spec

    def calculate_mfcc(self, y, beat_frames):
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

        self.mfccs.append(beat_mfcc_delta)
        return mfcc

    def calculate_tempogram(self, y, beat_frames):
        oenv = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.sr, hop_length=self.hop_length)
        beat_tempogram = librosa.util.sync(tempogram, beat_frames)
        self.tempograms.append(beat_tempogram)

        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=self.sr, hop_length=self.hop_length)[0]
        self.tempos.append(tempo)

        return tempogram

    def calculate_chromagram(self, y, beat_frames):
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        beat_chroma = librosa.util.sync(chroma_cq, beat_frames, aggregate=np.median)

        self.chromagrams.append(beat_chroma)
        return chroma_cq

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

    def plot_mel_spec(self, data, genre):
        title = f'Mel Spectrogram Example for {genre.capitalize()} Genre'
        path = f'{self.output_graph_path}/mel_spec/{genre}.png'
        data = librosa.power_to_db(np.abs(data), ref=np.max)

        self.plot_graph(data, title, path, x_axis='time', y_axis='mel', ax_format='%+2.0f dB')

    def plot_mfcc(self, data, genre):
        title = f'MFCC Example for {genre.capitalize()} Genre'
        path = f'{self.output_graph_path}/mfcc/{genre}.png'

        self.plot_graph(data, title, path, x_axis='time', y_axis='linear')

    def plot_tempogram(self, data, idx, genre):
        title = f'Tempogram Example for {genre.capitalize()} Genre'
        path = f'{self.output_graph_path}/tempogram/{genre}.png'

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(title)
        librosa.display.specshow(data, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='tempo', cmap='magma', ax=ax)
        ax.axhline(self.tempos[idx], color='w', linestyle='--', alpha=1, label=f'Estimated tempo={self.tempos[idx]}')
        ax.legend(loc='upper right')

        fig.savefig(path, bbox_inches='tight', pad_inches=0.25)
        fig.show()
        fig.clf()

    def plot_chromagram(self, data, genre):
        title = f'Chromagram Example for {genre.capitalize()} Genre'
        path = f'{self.output_graph_path}/chromagram/{genre}.png'

        self.plot_graph(data, title, path, x_axis='time', y_axis='chroma')

    # =================================================================
    #                         Preprocess
    # =================================================================
    def load_features(self):
        if len(self.features) > 0:
            return self.features
        else:
            for genre_idx, genre in enumerate(self.genres):
                for idx in range(50):
                    idx_name = f'00{idx}' if idx <= 9 else f'0{idx}'
                    self.labels.append(genre)

                    if glob(f'output/features/{genre}/{idx_name}.npy'):
                        self.features.append(utils.read_feat_from_files(genre, idx_name))

                    else:
                        if len(self.audios) == 0:
                            self.load_audios()

                        # pad features to max
                        mel_spec = utils.pad_data(self.mel_specs[idx + genre_idx * 50], self.max_dim)
                        mfcc = utils.pad_data(self.mfccs[idx + genre_idx * 50], self.max_dim)
                        tempogram = utils.pad_data(self.tempograms[idx + genre_idx * 50], self.max_dim)
                        chromagram = utils.pad_data(self.chromagrams[idx + genre_idx * 50], self.max_dim)

                        feat = np.concatenate((mel_spec, mfcc, tempogram, chromagram), axis=0)

                        if idx == 0:
                            print('--- Combining Features ---')
                            print(f'Padded all features to {self.max_dim}')
                            print(f'> mel_spec: {mel_spec.shape}')
                            print(f'> mfcc: {mfcc.shape}')
                            print(f'> tempogram: {tempogram.shape}')
                            print(f'> chromagram: {chromagram.shape}')
                            print(f'> combined feat: {feat.shape}')

                        utils.write_feat_to_files(genre, idx_name, feat)
                        self.features.append(feat)

            print('Loaded all features for all audios')
            return self.features
