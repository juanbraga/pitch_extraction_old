# -*- coding: utf-8 -*-

import vamp
import librosa
import matplotlib.pyplot as plt
import copy

<<<<<<< HEAD
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer_mono.wav'
audio_file = "C:/Users/Juan/Documents/FingMaestria/tesis/traditional-dataset-repo/syrinx/fragments/syrinx_third_fragment_rhodes_mono.wav"
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin_mono.wav'
=======
#audio_file = '../traditional_dataset/density/fragments/density_third_fragment_zoon.wav'
audio_file = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes_mono.wav'
>>>>>>> c88c007b67d4f2076e26e39b2b189ac19e000e46
#audio_file = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet_mono.wav'
audio, sr = librosa.load(audio_file, sr=44100, mono=True)

#%%
# parameter values are specified by providing a dicionary:
params = {"minfqr": 100.0, "maxfqr": 2350.0, "voicing": 0.9, "minpeaksalience": 0.0}
data = vamp.collect(audio, sr, "mtg-melodia:melodia", parameters=params)
hop, melody_melodia = data['vector']

#melody_librosa, magnitudes = librosa.piptrack(audio, sr=sr, hop_length=128)
#print(hop)
#print(melody)

import numpy as np
timestamps = 8 * 128/44100.0 + np.arange(len(melody_melodia)) * (128/44100.0)


melody_hz = copy.deepcopy(melody_melodia)
melody_hz[melody_melodia<=0] = None
plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_hz)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (cents relative to 55 Hz)')
plt.show()

#%%
import melosynth as ms
ms.melosynth_pitch(melody_melodia, 'melosynth.wav', fs=44100, nHarmonics=1, square=True, useneg=False) 

#%%
from pydub import AudioSegment
sound1 = AudioSegment.from_file(audio_file)
sound1 = sound1.pan(+1)

sound2 = AudioSegment.from_file("melosynth.wav")
sound2 = sound2.apply_gain(-10)
sound2 = sound2.pan(-1)

combined = sound1.overlay(sound2)
combined.export("combined.wav", format='wav')

