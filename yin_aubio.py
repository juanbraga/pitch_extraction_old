#! /usr/bin/env python

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from aubio import pitch, freqtomidi
import copy

#fragment = '../traditional_dataset/density/fragments/density_fifth_fragment_beauregard'
#fragment = '../traditional_dataset/density/fragments/density_third_fragment_zoon'
#fragment = '../traditional_dataset/density/fragments/density_second_fragment_zoon'
#fragment = '../traditional_dataset/density/fragments/density_first_fragment_zoon'

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

fragment = '../traditional_dataset/sequenza/fragments/sequenza_first_fragment_robison'

audio_file = fragment + '_mono.wav'
gt_file = fragment + '.csv'

fs,audio = wav.read(audio_file)
audio = audio.astype('float32', copy=False)

win_s=2048
hop_s=512
tolerance = 0.2

pitch_o = pitch("yin", win_s, hop_s)
pitch_o.set_unit("Hz")
pitch_o.set_tolerance(tolerance)
pitch_o.set_silence(-1)

pitches = []
confidences = []

#%%

# total number of frames read
total_frames = len(audio)/hop_s
for i in range(0,total_frames-1):
    
    samples = audio[i*hop_s:i*hop_s+win_s]
    pitch = pitch_o(samples)[0]
#    pitch = int(round(pitch))
    confidence = pitch_o.get_confidence()
    if confidence < 0.8: pitch = 0.
#    #print "%f %f %f" % (total_frames / float(samplerate), pitch, confidence)
    pitches += [pitch]
    confidences += [confidence]

timestamps = np.arange(len(pitches)) * (hop_s/44100.0)

pitches = np.array(pitches)
melody_hz = copy.deepcopy(pitches)
melody_hz[pitches<=200] = None
melody_hz[pitches>2500] = None
pitches[pitches<=200] = 0
pitches[pitches>2500] = 0

#plt.figure(figsize=(18,6))
#plt.plot(timestamps, melody_hz)
#plt.xlabel('Time (s)')
#plt.ylabel('Frequency (cents relative to 55 Hz)')
#plt.show()

#%%
import melosynth as ms
ms.melosynth_pitch(pitches, 'melosynth.wav', fs=44100, nHarmonics=1, square=True, useneg=False) 

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
        melo = np.r_[melo,100]
    else:
        melo = np.r_[melo,frequency[notation.index(note)]]
    i=i+1

j=0
gt = np.empty([len(timestamps),],'float64')
for i in range(1,len(onset)):
    while (j<len(timestamps) and (timestamps[j])>=onset[i-1] and (timestamps[j])<=onset[i]):
        gt[j]=melo[i-1]
        j=j+1

plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_hz)
plt.plot(timestamps, gt)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
#plt.yscale('log')
#plt.yticks(frequency, notation)
#plt.ylim(ymax = 2350 , ymin = 100)
plt.axis('tight')
plt.grid()
plt.show()
            
ms.melosynth_pitch(gt, 'melosynth_gt.wav', fs=44100, nHarmonics=1, square=True, useneg=False) 

sound1 = AudioSegment.from_file(audio_file)
sound1 = sound1.pan(+1)

sound2 = AudioSegment.from_file("melosynth_gt.wav")
sound2 = sound2.apply_gain(-14)
sound2 = sound2.pan(-1)

combined = sound1.overlay(sound2)
combined.export("combined_gt.wav", format='wav')