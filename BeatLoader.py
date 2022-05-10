import essentia.standard as es
import matplotlib.pyplot as plt


class BeatLoader:
    def __init__(self, path, print_logs):
        self.path = path
        self.print_logs = print_logs
        self.audio = None

        self.bpm = -1
        self.beats = []
        self.beats_intervals = []

        # histogram params (6-dim)
        self.beat_hist = []
        self.first_weight = 0
        self.second_weight = 0
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

        if self.print_logs:
            print(f'----- Rhythmic Features Starts -----')
            print("BPM:", self.bpm)

    # =================================================================
    #                           Calculations
    # =================================================================

    def calculate_beat_histogram(self):
        self.first_bpm, self.first_weight, peak1_spread, \
            self.second_bpm, self.second_weight, peak2_spread, \
            self.beat_hist = \
            es.BpmHistogramDescriptors()(self.beats_intervals)

    def get_rhythmic_feats(self):
        if self.print_logs:
            print("First amplitude (A0): %0.1f amp" % self.first_weight)
            print("Second amplitude (A1): %0.1f amp" % self.second_weight)
            print("A1/A0 (RA): %0.1f " % self.relative_amp)
            print("First peak (P1): %0.1f bpm" % self.first_bpm)
            print("Second peak (P2): %0.1f bpm" % self.second_bpm)
            print(f'----- Rhythmic Features Ends -----')

        return self.first_weight, self.second_weight, self.relative_amp, self.first_bpm, self.second_bpm

    # =================================================================
    #                           Plotting
    # =================================================================

    def plot_wave_and_beat(self, title, path):
        plt.figure(figsize=(15, 6))
        plt.plot(self.audio)
        for beat in self.beats:
            plt.axvline(x=beat * 44100, color='red')
        plt.xlabel('Time (samples)')
        plt.title(title)

        plt.savefig(path, bbox_inches='tight', pad_inches=0.25)
        plt.show()
        plt.clf()

    def plot_beat_hist(self, title, path):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(title)
        ax.bar(range(len(self.beat_hist)), self.beat_hist, width=1)
        ax.set_xlabel('BPM')
        ax.set_ylabel('Frequency of occurrence')
        ax.set_xticks(range(40, 220, 20))
        ax.set_xticklabels(range(40, 220, 20))

        fig.show()
        fig.savefig(path, bbox_inches='tight', pad_inches=0.25)
        fig.clf()
