# create real-time audio stream from real-time emotion data

from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer, QSize

from gensound import WAV, Sine, Gain
import time
import pygame
import numpy as np
import math

from emili_core import time_since


class Sonifier(QObject):
    
    def __init__(self, start_time, speed, tonic, pipeline, end_session_event):
        super().__init__()
        self.start_time = start_time
        self.end_session_event = end_session_event
        self.interval = 1000//speed # refresh rate in ms
        self.tonic = tonic # tonic frequency in Hz
        self.overtone = [] # integer multiples of the tonic
        self.tone = [] # one tone for each emotion
        self.current_chord = None
        self.pipeline = pipeline
        self.num_channels = 16
        self.channel = 0 # current audio channel

        pygame.mixer.pre_init(44100, 16, self.num_channels, 4096) # (frequency, size, channels, buffer)
        #pygame.mixer.init(44100, -16, self.num_channels, 4096)
        pygame.init()
        pygame.mixer.set_num_channels(self.num_channels) 

        self.overtone.append(Sine(frequency=self.tonic, duration=1e3)*Gain(-9))
        for n in range(1,18): # overtones of self.tonic (For natural indexing, the tonic is repeated at indices 0 and 1.) 
            self.overtone.append(Sine(frequency=self.tonic*n, duration=1e2)*Gain(-9-n))
        
        anger_tone = self.overtone[7] + self.overtone[8] # anger
        disgust_tone = self.overtone[16] + self.overtone[17] # disgust
        fear_tone = self.overtone[11] + self.overtone[12] # fear
        happiness_tone = 0.5*self.overtone[4] + self.overtone[5] + 0.5*self.overtone[6] # happiness
        sadness_tone = 0.5*self.overtone[2] + self.overtone[3] + 0.5*self.overtone[4] # sadness
        surprise_tone = self.overtone[8]+self.overtone[9]+0.5*self.overtone[10] # surprise
        neutral_tone = self.overtone[1] + self.overtone[2] + self.overtone[4] # neutral

        self.tone = [anger_tone, disgust_tone, fear_tone, happiness_tone, sadness_tone, surprise_tone, neutral_tone]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(40) # calls step() when the timer rings

    def step(self):
        #call_time = time_since(self.start_time)
        #print(f"(sonifier.step) called at {call_time}")
        if len(self.pipeline.binned_time_series) == 0: 
            return # no data to sonify

        # play current chord
        if(self.current_chord is not None):
            pygame.mixer.Channel(self.channel).play(self.current_chord, fade_ms=10)
        
        # build new chord in a new channel
        self.channel += 1
        if self.channel >= self.num_channels:
            self.channel = 0
        #print(f"(sonifier.step) called at {call_time}, new channel: {self.channel}")
        new_score = self.pipeline.binned_time_series[-1][1]/1e6 # most recent emotion scores
        chord = self.tone[0]*new_score[0]
        for n in range(1,7):
            chord += self.tone[n]*new_score[n]
        filename = f"chord{self.channel}.wav"
        chord.export(filename) # 100ms chord
        self.current_chord = pygame.mixer.Sound(filename)
        #print(f"(sonifier.step) called at {call_time} finished at {time_since(self.start_time)}")

    def run(self):
        pass

    def stop(self):
        self.timer.stop()
        
