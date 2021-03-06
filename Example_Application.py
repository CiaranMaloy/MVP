import joblib
from scipy.io import wavfile
import Sheeran_Functions_mod as sf
import Test_Functions as test 
import os
import numpy as np
import time
import pickle as pkl

# names
file_loc = 'Files/'
test.make_folder(file_loc)
band = 5

# Load in pre-trained classifier function
filename = "Classifiers/Random_forest001.sav"  # classifier filename
classifier = joblib.load(filename)

# Load in wav file (album)
album_dir = '/Volumes/Audio_Drive/TagMix_Sessions/'
albums = test.get_albums(album_dir)
album = albums[15]

print(album)
# from album, get and sort tracks
album_tracks = test.get_tracks_mp3(album)  # no for loop right now
album_tracks = test.sort_album_tracks(album_tracks)

save_track_lengths = []
audio_full = []
album_samplerate = []

real_cut_point = 0
real_cut_points = [0]
for track in album_tracks: 
    try: 
        track = test.convert_mp3_file_to_wav(track)
    except: 
        print('couldnt convert to mp3')
        break

    track_name, ext = os.path.splitext(track)
    
    if ext == '.wav':
        fs, audio = wavfile.read(track)
        track_length_s = round(len(audio)/fs)
    else: 
        print('{} is not a wavfile!!'.format(track_name))
        break

    audio = test.remove_padding_sterio(audio)
    audio = test.preprocess(audio)

    save_track_lengths.append(track_length_s)
    audio_full.extend(audio)
    album_samplerate.append(fs) 

    real_cut_point += track_length_s
    real_cut_points.append(real_cut_point)
print('Done make mix loop')
live_set = np.array(audio_full)


# Set output file location
output_folder_name = "Output"
sf.make_folder(output_folder_name)

# set parameters
buffer_length = 2  # s
n_noise_thresh = 1  # Number of 500ms instances of noise within a buffer to classify as a new song
min_track_length = 10  # s - Ignore track's detected less than this, useful for getting rid of waffle.
                       # also if you don't set a minimum then long sections of conversation get split up, resulting in
                       # hundereds of exported files.... its a prototype, what do you expect?
rms_threshold = 0.08

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
    predictions, log_fs, pred_bool = sf.get_classification(input_buffer, fs, classifier, rms_threshold, verbose=True, tic=tic)

    # Makes Decision based on predictions, simple - if there are 5 instances of noise, then its a new song
    # returns, cut_bool, which is True or False depending on the decision of new track or not
    # and cut_point, which returns False when there is no new section,
    # and the cut point in seconds relative to the beginning of the buffer.
    cut_bool, cut_point = sf.make_decision(predictions, log_fs, pred_bool, n_noise_thresh=n_noise_thresh)

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

# compare cut points -- 
cut_points = np.array(cut_points)/fs
print('Pred Cut Points: {}'.format(cut_points))
print('Real Cut Points: {}'.format(real_cut_points))
print('Done')

total_true, total_flase, true_pos, false_pos = test.comparison(real_cut_points, cut_points, band)

print('********* Album Done **********')
print('*******************************')

save_dict = {
    'Album Name': album, # name
    'Total True': total_true, # number of real cut points (target)
    'True Pos': true_pos, # number of correct points
    'False Pos': false_pos, # number of false poits
    'True Cut Points': real_cut_points, # cut points taken from album splits 
    'Predicted Cut Points': cut_points, # predicted cut points
}
file_head = test.path_leaf(album)
test.make_folder(file_loc + file_head)
print('Save Album ' + file_head)
dict_name = file_loc + file_head + '/testInfo.pkl' 
f = open(dict_name,"wb")
pkl.dump(save_dict,f)
f.close()
    
print('*** __ Test Done ___ ***')

print('cut points: {}'.format(cut_points))
print('Done')
