import joblib
from scipy.io import wavfile
import Sheeran_Functions as sf
import os
import numpy as np
import time

# Load in pre-trained classifier function
filename = "Classifiers/Random_forest001.sav"  # classifier filename
classifier = joblib.load(filename)

# Load in wav file
folder = os.getcwd() + "/Place_Wav_or_Mp3_Files/"
wav_files = sf.process_wav_or_mp3_files(folder)
wav_file = wav_files[0]

# Set output file location
output_folder_name = "Output"
sf.make_folder(output_folder_name)

# process live set
fs, live_set = wavfile.read(wav_file)
live_set = sf.remove_padding(live_set)

# set parameters
buffer_length = 10  # s
n_noise_thresh = 7  # Number of 500ms instances of noise within a buffer to classify as a new song
min_track_length = 30  # s - Ignore track's detected less than this, useful for getting rid of waffle.
                       # also if you don't set a minimum then long sections of conversation get split up, resulting in
                       # hundereds of exported files.... its a prototype, what do you expect?

# calculate buffers in signal
n_buffers_in_signal = int(np.floor(len(live_set) / (buffer_length * fs)))
tic = time.perf_counter()

hold = []
cut_points = []
n_cut = 0
n_tracks = 0
for buffer in range(n_buffers_in_signal):

    # select section
    a = int(buffer * buffer_length * fs)
    b = int((buffer + 1) * buffer_length * fs)
    input_buffer = live_set[a:b]

    # Make prediction on sections in buffer,
    # Takes, classifier object, input buffer waveform, sample frequency, expects 44.1kHz audio, can handle 22.05kHz
    predictions, log_fs = sf.get_classification(input_buffer, fs, classifier, verbose=True, tic=tic)

    # Makes Decision based on predictions, simple - if there are 5 instances of noise, then its a new song
    # returns, cut_bool, which is True or False depending on the decision of new track or not
    # and cut_point, which returns False when there is no new section,
    # and the cut point in seconds relative to the beginning of the buffer.
    cut_bool, cut_point = sf.make_decision(predictions, log_fs, n_noise_thresh=n_noise_thresh)

    # print things that matter
    print('Buffer {} of {}'.format(buffer, n_buffers_in_signal))
    print('Predictions: {}'.format(predictions))
    print('Cut?: {}'.format(cut_bool))
    print('Cut Point: {}'.format(cut_point))
    print('-----------------------------------------------------')

    #
    if cut_bool:
        true_cut_point = int(a + cut_point * fs)
        cut_points.append(true_cut_point)

        # array must be greater than two so that there are two points to get audio from between
        if len(cut_points) > 2:
            point_a = cut_points[n_cut]
            point_b = cut_points[n_cut + 1]
            n_cut += 1
            track_length = (point_b - point_a) / fs

            # writes tracks as a proof of concept
            if track_length > min_track_length:
                print('-----------------------------------------------------')
                n_tracks += 1
                filename = output_folder_name + "/track_{}.wav".format(n_tracks)
                print('Writing: {}'.format(filename))
                wavfile.write(filename, fs, live_set[point_a:point_b])
                print('Done')
                print('-----------------------------------------------------')

print('cut points: {}'.format(cut_points))
print('Done')
