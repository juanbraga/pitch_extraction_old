# -*- coding: utf-8 -*-

import vamp
import librosa
import matplotlib.pyplot as plt
import copy


fragment = '../traditional_dataset/density/fragments/density_first_fragment_zoon'

#fragment = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin'

#fragment = '../traditional_dataset/allemande/fragments/allemande_second_fragment_gerard'
#fragment = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet'
#fragment = '../traditional_dataset/allemande/fragments/allemande_third_fragment_rampal'
#fragment = '../traditional_dataset/allemande/fragments/allemande_fourth_fragment_larrieu'
#fragment = '../traditional_dataset/allemande/fragments/allemande_fifth_fragment_preston'

audio_file = fragment + '_mono.wav'
gt_file = fragment + '.csv'

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

#%%
import csv

cr = csv.reader(open(gt_file,"rb"))
      
onset=[]
notes=[]

for row in cr:

    onset.append(row[0]) 
    notes.append(row[1])

onset = np.array(onset, 'float64')

cr = csv.reader(open("./note_convertion.csv","rb"))
      
notation=[]
frequency=[]

for row in cr:

    notation.append(row[0]) 
    frequency.append(row[1])

frequency = np.array(frequency, 'float64')

#plt.figure(figsize=(18,6))
#plt.plot(timestamps, melody_hz)
#plt.xlabel('Time (s)')
#plt.ylabel('Frequency')
#plt.show()

i=0
melo = np.empty([0,])
for note in notes:
    if note=='0':
        melo = np.r_[melo,0]
    else:
        melo = np.r_[melo,frequency[notation.index(note)]]
    i=i+1
    

#%%

j=0
gt = np.empty([len(timestamps),],'float64')
for i in range(1,len(onset)):
    while (j<len(timestamps) and (timestamps[j]-(8*128/44100.0))>=onset[i-1] and (timestamps[j]-(8*128/44100.0))<=onset[i]):
        gt[j]=melo[i-1]
        j=j+1

plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_hz)
plt.plot(timestamps, gt)
plt.xlabel('Time (s)')
plt.ylabel('Frequency')
plt.show()
            
ms.melosynth_pitch(gt, 'melosynth_gt.wav', fs=44100, nHarmonics=1, square=True, useneg=False) 

sound1 = AudioSegment.from_file(audio_file)
sound1 = sound1.pan(+1)

sound2 = AudioSegment.from_file("melosynth_gt.wav")
sound2 = sound2.apply_gain(-10)
sound2 = sound2.pan(-1)

combined = sound1.overlay(sound2)
combined.export("combined_gt.wav", format='wav')