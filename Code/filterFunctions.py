#!/usr/bin/env python

from scipy.signal import butter, lfilter
    
def butter_bandpass_filter(data, lowcut, highcut, fs, btype='band', order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, btype, order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, btype, order=5):
    """btype can be either 'band' or 'bandstop' for a band pass or band stop filter """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype)
    return b, a