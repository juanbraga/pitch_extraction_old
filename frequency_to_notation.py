# -*- coding: utf-8 -*-

#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import csv
import librosa as lr

from aubio import pitch, freqtomidi
import copy
import tradataset as td

def pitch_extraction(audio, fs, win, hop):

    audio = audio.astype('float32', copy=False)
    
    win_s=win
    hop_s=hop
    tolerance = 0.5
    
    pitch_o = pitch("yin", win_s, hop_s, fs)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(tolerance)
    
    pitches = []
    confidences = []
    
    # total number of frames read
    total_frames = len(audio)/win_s
    for i in range(0,total_frames-1):
        
        samples = audio[i*win_s:(i+1)*win_s]
        p = pitch_o(samples)[0]
        #pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        #if confidence < 0.8: pitch = 0.
        #print "%f %f %f" % (total_frames / float(samplerate), pitch, confidence)
        pitches += [p]
        confidences += [confidence]
    
    timestamps = np.arange(len(pitches)) * float(hop)/fs
    
    pitches = np.array(pitches)
    melody_hz = copy.deepcopy(pitches)
    melody_hz[pitches<=0] = None
    melody_hz[pitches>1200] = None

    melonotes = lr.hz_to_midi(melody_hz);
    int_melonotes=np.round(melonotes) 
    
    return int_melonotes, timestamps
    

#if __name__ == "__main__":  
#
#    ldataset = td.load_list()    
#    fragment = ldataset[3]    
#       
#    audio_file = fragment + '_mono.wav'
#    gt_file = fragment + '.csv'
#    
#    fs,audio = wav.read(audio_file)
#    audio = audio.astype('float32', copy=False)
#    
#    win_s=2048
#    hop_s=0
#    tolerance = 0.5
#    
#    pitch_o = pitch("yin", win_s, hop_s, fs)
#    pitch_o.set_unit("Hz")
#    pitch_o.set_tolerance(tolerance)
#    
#    pitches = []
#    confidences = []
#    
#    # total number of frames read
#    total_frames = len(audio)/win_s
#    for i in range(0,total_frames-1):
#        
#        samples = audio[i*win_s:(i+1)*win_s]
#        pitch = pitch_o(samples)[0]
#        #pitch = int(round(pitch))
#        confidence = pitch_o.get_confidence()
#        #if confidence < 0.8: pitch = 0.
#        #print "%f %f %f" % (total_frames / float(samplerate), pitch, confidence)
#        pitches += [pitch]
#        confidences += [confidence]
#    
#    timestamps = np.arange(len(pitches)) * (2048/44100.0)
#    
#    pitches = np.array(pitches)
#    melody_hz = copy.deepcopy(pitches)
#    melody_hz[pitches<=0] = None
#    melody_hz[pitches>1200] = None
#        
#    cr = csv.reader(open(gt_file,"rb"))
#          
#    onset=[]
#    notes=[]
#    
#    for row in cr:
#    
#        onset.append(row[0]) 
#        notes.append(row[1])
#    
#    onset = np.array(onset, 'float64')
#    
#    cr = csv.reader(open("./note_convertion.csv","rb"))
#          
#    notation=[]
#    frequency=[]
#    
#    for row in cr:
#    
#        notation.append(row[0]) 
#        frequency.append(row[1])
#    
#    frequency = np.array(frequency, 'float64')
#    
#    i=0
#    melo = np.empty([0,])
#    for note in notes:
#        if note=='0':
#            melo = np.r_[melo,0]
#        else:
#            melo = np.r_[melo,frequency[notation.index(note)]]
#        i=i+1
#        
#    j=0
#    gt = np.empty([len(timestamps),],'float64')
#    for i in range(1,len(onset)):
#        while (j<len(timestamps) and (timestamps[j])>=onset[i-1] and (timestamps[j])<=onset[i]):
#            gt[j]=melo[i-1]
#            j=j+1
#    
#    plt.figure(figsize=(18,6))
#    plt.plot(timestamps, melody_hz)
#    plt.plot(timestamps, gt)
#    plt.xlabel('Time (s)')
#    plt.ylabel('Notes')
#    plt.yscale('log')
#    plt.yticks(frequency, notation)
#    plt.ylim(ymax = 2350 , ymin = 246)
#    plt.axis( )
#    plt.show()
#    
#    #%%
#    midigt = lr.hz_to_midi(gt);    
#    melonotes = lr.hz_to_midi(melody_hz);
#    int_melonotes=np.round(melonotes)   
#    
#    plt.figure()
#    plt.plot(timestamps,int_melonotes,'.-',color='blue', lw=0.7)
#    plt.plot(timestamps,midigt,'green', lw=1.4)
#    plt.plot(timestamps,melonotes, 'red', lw=0.3)
#    plt.fill_between(timestamps, midigt, int_melonotes, facecolor='cyan', label='diference', alpha=0.2)
#    plt.grid(b=True, which='major', color='black', axis='y', linestyle='-')
#    plt.grid(b=True, which='minor', color='black', axis='y', linestyle='-', alpha=0.3)
#    plt.minorticks_on()
#    