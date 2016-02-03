# -*- coding: utf-8 -*-

import vamp
import librosa
import matplotlib.pyplot as plt
import copy

#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer_mono.wav'
audio_file = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin_mono.wav'
#audio_file = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet_mono.wav'
audio, sr = librosa.load(audio_file, sr=44100, mono=True)

# parameter values are specified by providing a dicionary:
params = {"minfqr": 100.0, "maxfqr": 2350.0, "voicing": 0.9, "minpeaksalience": 0.0}
data = vamp.collect(audio, sr, "mtg-melodia:melodia", parameters=params)
hop, melody = data['vector']
#print(hop)
#print(melody)

import numpy as np
timestamps = 8 * 128/44100.0 + np.arange(len(melody)) * (128/44100.0)

#%%
# Melodia returns unvoiced (=no melody) sections as negative values. So by default, we get:
plt.figure(figsize=(18,6))
plt.plot(timestamps, melody)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

# A clearer option is to get rid of the negative values before plotting
melody_pos = copy.deepcopy(melody)
melody_pos[melody<=0] = None
plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_pos)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

# Finally, you might want to plot the pitch sequence in cents rather than in Hz. 
# This especially makes sense if you are comparing two or more pitch sequences 
# to each other (e.g. comparing an estimate against a reference).
melody_cents = 1200*np.log2(melody/55.0)
melody_cents[melody<=0] = None
plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_cents)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (cents relative to 55 Hz)')
plt.show()

#%%
import melosynth as ms
ms.melosynth_pitch(melody, 'melosynth.wav', fs=44100, nHarmonics=1, square=True, useneg=False) 

#%%
from pydub import AudioSegment
sound1 = AudioSegment.from_file(audio_file)
sound1 = sound1.pan(+1)

sound2 = AudioSegment.from_file("melosynth.wav")
sound2 = sound2.apply_gain(-10)
sound2 = sound2.pan(-1)

combined = sound1.overlay(sound2)
combined.export("combined.wav", format='wav')

