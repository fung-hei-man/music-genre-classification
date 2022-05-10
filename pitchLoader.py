import essentia.standard as es
import matplotlib.pyplot as plt


class PitchLoader:
    def __init__(self, path, sr, print_logs):
        self.path = path
        self.sr = sr
        self.print_logs = print_logs
        self.audio = None

        self.pitches = []
        self.unfolded_hist = []
        self.folded_hist = []

    def load_pitches(self):
        # Loading audio file
        self.audio = es.MonoLoader(filename=self.path)()
        self.audio = es.EqualLoudness(sampleRate=self.sr)(self.audio)

        pitch_extractor = es.PitchMelodia()
        pitch, pitch_conf = pitch_extractor()

        print(pitch)
