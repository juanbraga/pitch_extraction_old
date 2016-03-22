# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:44:39 2016

@author: Juan
"""

from pydub import AudioSegment

sound1 = AudioSegment.from_file('Voz Amba.wav')
sound1 = sound1.pan(+1)

sound2 = AudioSegment.from_file("amba_melosynth.wav")
sound2 = sound2.apply_gain(-10)
sound2 = sound2.pan(-1)

combined = sound1.overlay(sound2)
combined.export("combined_gt.wav", format='wav')