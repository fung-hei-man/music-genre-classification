import essentia
import essentia.standard as es
import matplotlib.pyplot as plt


class BeatLoader:
    def __init__(self, path):
        self.path = path
        self.audio = None

        self.bpm = -1
        self.beats = []
        self.beats_intervals = []

        # histogram params (6-dim)
        self.beat_hist = []
        self.nom_first_amp = 0
        self.nom_second_amp = 0
        self.relative_amp = 0
        self.first_bpm = 0
        self.second_bpm = 0
        self.sum = 0

    def load_beats(self):
        # Loading audio file
        self.audio = es.MonoLoader(filename=self.path)()

        # Compute beat positions and BPM
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        self.bpm, self.beats, _, _, self.beats_intervals = rhythm_extractor(self.audio)

        print("BPM:", self.bpm)
        print("Beat positions (sec.):", self.beats)

    # =================================================================
    #                           Calculations
    # =================================================================

    def calculate_beat_histogram(self):
        self.first_bpm, peak1_weight, peak1_spread, \
            self.second_bpm, peak2_weight, peak2_spread, \
            self.beat_hist = \
            es.BpmHistogramDescriptors()(self.beats_intervals)

        self.sum = sum(self.beat_hist)
        self.nom_first_amp = peak1_weight / self.sum
        self.nom_second_amp = peak2_weight / self.sum

    def get_rhythmic_feats(self):
        print("First relative amplitude (A0): %0.1f bpm" % self.nom_first_amp)
        print("Second relative amplitude (A1): %0.1f bpm" % self.nom_second_amp)
        print("A1/A0 (RA): %0.1f bpm" % self.relative_amp)
        print("First peak (P1): %0.1f bpm" % self.first_bpm)
        print("Second peak (P2): %0.1f bpm" % self.second_bpm)
        print("Histgram sum (SUM): %0.1f bpm" % self.sum)

        return self.nom_first_amp, self.nom_second_amp, self.relative_amp, self.first_bpm, self.second_bpm, self.sum

    # =================================================================
    #                           Plotting
    # =================================================================

    def plot_wave_and_beat(self, title, path):
        plt.rcParams['figure.figsize'] = (15, 6)
        plt.plot(self.audio)
        for beat in self.beats:
            plt.axvline(x=beat * 44100, color='red')
        plt.xlabel('Time (samples)')
        plt.title(title)

        plt.show()
        plt.savefig(path)
        plt.clf()

    def plot_beat_hist(self, title, path):
        fig, ax = plt.subplots()
        ax.bar(range(len(self.beat_hist)), self.beat_hist, width=1)
        ax.set_xlabel('BPM')
        ax.set_ylabel('Frequency of occurrence')
        plt.title(title)
        ax.set_xticks([20 * x + 0.5 for x in range(int(len(self.beat_hist) / 20))])
        ax.set_xticklabels([str(20 * x) for x in range(int(len(self.beat_hist) / 20))])

        plt.show()
        plt.savefig(path)
        plt.clf()
