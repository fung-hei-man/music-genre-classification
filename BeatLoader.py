import essentia
import essentia.standard as es
import matplotlib.pyplot as plt


class BeatLoader:
    def __init__(self, path):
        self.path = path
        self.bpm = -1
        self.beats = []
        self.beats_interval = []

    def load_beats(self):
        # Loading audio file
        audio = es.MonoLoader(filename=self.path)()

        # Compute beat positions and BPM
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        print("BPM:", bpm)
        print("Beat positions (sec.):", beats)
        print("Beat estimation confidence:", beats_confidence)

    def plot_wave_and_beat(self):
        # plt.rcParams['figure.figsize'] = (15, 6)
        # plt.plot(audio)
        # for beat in self.beats:
        #     plt.axvline(x=beat * 44100, color='red')
        # plt.xlabel('Time (samples)')
        # plt.title("Audio waveform and the estimated beat positions")
        # show()