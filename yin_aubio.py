#! /usr/bin/env python

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from aubio import pitch, freqtomidi
import copy

#audio_file='melosynth.wav'
#audio_file = '../traditional_dataset/density/fragments/density_third_fragment_zoon_mono.wav'
audio_file = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes_mono.wav'
#audio_file = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet_mono.wav'
fs,audio = wav.read(audio_file)
audio = audio.astype('float32', copy=False)

win_s=2048
hop_s=1024
tolerance = 0

pitch_o = pitch("yin", win_s, hop_s, fs)
pitch_o.set_unit("Hz")
pitch_o.set_tolerance(tolerance)

pitches = []
confidences = []

#%%

# total number of frames read
total_frames = len(audio)/win_s
for i in range(0,total_frames-1):
    
    samples = audio[i*win_s:(i+1)*win_s]
    pitch = pitch_o(samples)[0]
    #pitch = int(round(pitch))
    confidence = pitch_o.get_confidence()
    #if confidence < 0.8: pitch = 0.
    #print "%f %f %f" % (total_frames / float(samplerate), pitch, confidence)
    pitches += [pitch]
    confidences += [confidence]

timestamps = np.arange(len(pitches)) * (128/44100.0)

pitches = np.array(pitches)
melody_hz = copy.deepcopy(pitches)
melody_hz[pitches<=0] = None
melody_hz[pitches>1200] = None
plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_hz)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (cents relative to 55 Hz)')
plt.show()

#
##print pitches
#from numpy import array, ma
#import matplotlib.pyplot as plt
#from demo_waveform_plot import get_waveform_plot, set_xlabels_sample2time
#
#skip = 1
#
#pitches = array(pitches[skip:])
#confidences = array(confidences[skip:])
#times = [t * hop_s for t in range(len(pitches))]
#
#fig = plt.figure()
#
#ax1 = fig.add_subplot(311)
#ax1 = get_waveform_plot(audio_file, samplerate = fs, block_size = hop_s, ax = ax1)
#plt.setp(ax1.get_xticklabels(), visible = False)
#ax1.set_xlabel('')
#
#def array_from_text_file(filename, dtype = 'float'):
#    import os.path
#    from numpy import array
#    filename = os.path.join(os.path.dirname(__file__), filename)
#    return array([line.split() for line in open(filename).readlines()],
#        dtype = dtype)
#
#ax2 = fig.add_subplot(312, sharex = ax1)
#import sys, os.path
#ground_truth = os.path.splitext(filename)[0] + '.f0.Corrected'
#if os.path.isfile(ground_truth):
#    ground_truth = array_from_text_file(ground_truth)
#    true_freqs = ground_truth[:,2]
#    true_freqs = ma.masked_where(true_freqs < 2, true_freqs)
#    true_times = float(samplerate) * ground_truth[:,0]
#    ax2.plot(true_times, true_freqs, 'r')
#    ax2.axis( ymin = 0.9 * true_freqs.min(), ymax = 1.1 * true_freqs.max() )
## plot raw pitches
#ax2.plot(times, pitches, '.g')
## plot cleaned up pitches
#cleaned_pitches = pitches
##cleaned_pitches = ma.masked_where(cleaned_pitches < 0, cleaned_pitches)
##cleaned_pitches = ma.masked_where(cleaned_pitches > 120, cleaned_pitches)
#cleaned_pitches = ma.masked_where(confidences < tolerance, cleaned_pitches)
#ax2.plot(times, cleaned_pitches, '.-')
##ax2.axis( ymin = 0.9 * cleaned_pitches.min(), ymax = 1.1 * cleaned_pitches.max() )
##ax2.axis( ymin = 55, ymax = 70 )
#plt.setp(ax2.get_xticklabels(), visible = False)
#ax2.set_ylabel('f0 (midi)')
#
## plot confidence
#ax3 = fig.add_subplot(313, sharex = ax1)
## plot the confidence
#ax3.plot(times, confidences)
## draw a line at tolerance
#ax3.plot(times, [tolerance]*len(confidences))
#ax3.axis( xmin = times[0], xmax = times[-1])
#ax3.set_ylabel('condidence')
#set_xlabels_sample2time(ax3, times[-1], samplerate)
#plt.show()
##plt.savefig(os.path.basename(filename) + '.svg')