import os
import random
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import glob

# files and stuff
def make_folder(item):
    if not os.path.exists(item):
        print(item + ' Does Not Exist, Creating')
        os.mkdir(item)

# return n number wav files from list
def pick_random_wav_files(file_list, song_lengths):
    
    n_files = len(song_lengths)
    N = len(file_list)

    files = []
    i = 0
    while True:
        file_one = file_list[randint(N)]
            
        ext_one = os.path.splitext(file_one)
        ext = ext_one[1]
        if ext == '.mp3':
            file_one = convert_mp3_file_to_wav(file_one)
            ext = '.wav'
        elif ext == '.wav':
            file_one = file_one
        else: 
            print(file_one + ' -- is not appropriate')
        
        if ext == '.wav':
            files.append(file_one)
            i += 1
            if i == n_files:
                break

    return files


def convert_mp3_file_to_wav(path):

    if not os.path.exists(path):
        print(' Does Not Exist')

    #-- im no longer entirely sure what all this means --
    name = os.path.splitext(os.path.basename(path))
    name = name[0]
    file_name = os.path.split(path)
    folder_name = file_name[0]
    # --

    wav_name = os.path.join(folder_name, name + ".wav")
    if not os.path.isfile(wav_name):
        print('Converting: ' + name)
        mp3_name = os.path.join(folder_name, name + ".mp3")
        sound = AudioSegment.from_mp3(mp3_name)
        sound.export(wav_name, format="wav")
        print(name + ' done')
    else:
        print('{} - already exists!'.format(file_name[1]))
    
    return wav_name


def randint(x):
    return int(round(random.uniform(0,1) * x))

# get a list of all files in a directory
def get_files(music_dir):
    files = absoluteFilePaths(music_dir)

    file_list = []
    for file in files:
        file_list.append(file)
    return file_list

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

# set known cut points 
# --
def song_lengths(mix_length, min_song_length, max_song_length, verbose=False):
    song_lengths = []
    i = 0
    track = 0
    while True:
        length = randint(max_song_length)
        if length > min_song_length:
            track += length 

            if track > mix_length:
                break

            i += 1
            song_lengths.append(length)

    # add the end point on
    song_lengths.append(mix_length - sum(song_lengths))
    if sum(song_lengths) > mix_length:
        print('Why is this going wrong?')

    # --
    if verbose:
        print('Points: {}'.format(song_lengths))
        print('Track: {}'.format(track))
        print('Mix Length: {}'.format(mix_length))
        print('Sum Cut Points: {}'.format(sum(song_lengths)))

    return song_lengths    

# get them for reference
def get_cut_points(song_lengths):
    cut_points = [0]
    cut_point = 0
    for length in song_lengths:
        cut_point += length 
        cut_points.append(cut_point)
    return cut_points

# get random T seconds of audio from section 
def random_audio_section(file, seconds):
    
    fs, full_audio = wavfile.read(file)
    
    if len(full_audio) > seconds * fs:
        a = randint(len(full_audio) - seconds*fs)
        b = int(a + seconds*fs)
        time_limited_audio = full_audio[a:b]
        
    elif len(full_audio) < seconds*fs:
        time_limited_audio = False
        print('there is not enough data in file:')
        print(file)
    
    return time_limited_audio, fs

# take each audiosection and put it in an array
def array_of_audio_sections(wav_files, song_lengths, fade):
    n_files = len(song_lengths)    

    audio_array = [] # should reall find a way of pre-allocating this 
    fs_array = []
    for i in range(n_files):
        if i == 0 or i == n_files-1:
            time = int(round(song_lengths[i] + fade/2))
            audio, fs = random_audio_section(wav_files[i], time)
        else: 
            time = int(round(song_lengths[i] + fade))
            audio, fs = random_audio_section(wav_files[i], time)

        audio = remove_padding_sterio(audio)
        audio = preprocess(audio)
        audio_array.append(audio)
        fs_array.append(fs) # this is an array of audio files, that need to be mixed/faded into one another

    return audio_array, fs_array

# mono preprocessing
def preprocess(signal):
    signal = make_mono(signal)
    signal = normalise(signal)
    return signal   

def make_mono(data):
    data = np.array(data, dtype=np.float32)
    try:
        if len(data[0]) == 2:
            data = np.matrix.transpose(data)
            data = normalise(0.5 * data[0] + 0.5 * data[1])
    except:
        data = data
    return data

def normalise(x):
    x = x/np.abs(x).max()
    return x

def remove_padding_sterio(audio):
    offset = 0
    for i in audio:
        if i[0] == 0:
            offset += 1
        if i[0] != 0:
            break

    return audio[offset:]

# mixes all the parts together with a fade - hanning window used for fade
# lets be real here - everything should be the same sample rate, 44.1kHz ideally
# don't be weird and mix samplerates, just don't
def make_mix(audio_array, fs_array, fade):
    for i in range(len(fs_array)-1):
        if i == 0:
            mix = audio_array[i]

        t_fade = int(fade*fs_array[i])
        # end mix section file n
        end_mix_section = np.array(mix[-t_fade:])
        end_window = np.hanning(2*t_fade)[-t_fade:]
        end_mix_section = end_mix_section * end_window
        # start mix_section file n + 1
        start_mix_section = np.array(audio_array[i+1][:t_fade])
        start_window = np.hanning(2*t_fade)[:t_fade]
        start_mix_section = start_mix_section * start_window
        # superimpose secations
        mixed_section = preprocess(end_mix_section + start_mix_section)

        # see if this works before making more efficient
        mix_length = int(len(mixed_section))
        start_length = len(mix) - mix_length
        start_section = preprocess(mix[:start_length])
        end_section = preprocess(audio_array[i+1][mix_length:])
        mix = np.concatenate((start_section, mixed_section, end_section), axis=None)

    return mix, fs_array[0]

# -- ** ** ** ** ** ** ** ** 
# -- -- -- -- -- -- -- -- -- 
# -- Album Test Functions 
# -- -- -- -- -- -- -- -- -- 
# -- ** ** ** ** ** ** ** ** 

# get albums from album directory (album_dir) - returns list of album locations
def get_albums(album_dir):
    albums = []
    for root, dirs, _ in os.walk(album_dir, topdown=False):
        for name in dirs:
            albums.append(os.path.join(root, name + '/'))
    return albums

def get_tracks_mp3(album):
    tracks = absoluteFilePaths(album)
    out = []
    for track in tracks:
        track_name, ext = os.path.splitext(track)
        _, name = os.path.split(track_name)
        if ext == '.mp3' and name[0] != '.':
            out.append(track)
    return out

def sort_album_tracks(album_tracks):
    names = []
    for track in album_tracks:
        path, track_name = os.path.split(track)
        names.append(track_name)
    names.sort()
    tracks = []
    for name in names:
        tracks.append(path + '/' + name)
    return tracks

# comparison fuction to give True Pos and False Pos
def comparison(real_cut_points, pred_cut_points, band):
    # theres deffo a better way of doing this than the one im using!! 
    total = 0
    true_pos = 0  # which ones work 
    true_pos_extras = 0
    false_pos = 0  # which are extra

    # finds points that should be there but ony counts them once per band 
    for point in real_cut_points:
        trigger = 0
        for pred in pred_cut_points:
            if point-band < pred < point+band: 
                total += 1
                trigger += 1
                if trigger == 1:
                    true_pos += 1
                if trigger > 1:
                    true_pos_extras += 1

    # finds cut points outside of all bands 
    for pred in pred_cut_points: 
        trigger = 0
        for point in real_cut_points:
            if not point-band < pred < point+band: 
                trigger += 1
                if trigger == len(real_cut_points):
                    false_pos += 1

    total_true_cuts = len(real_cut_points)
    total_flase_cuts = len(pred_cut_points) - total_true_cuts
    print('True Pos: {}/{}'.format(true_pos, total_true_cuts))
    print('False Pos: {}'.format(false_pos))
    print('Extras for True Pos: {}'.format(true_pos_extras))

    return total_true_cuts, total_flase_cuts, true_pos, false_pos

# gets last item in pathname
def path_leaf(path):
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)
    
def seconds_to_minutes(seconds):
    min, sec = divmod(seconds, 60) 
    hour, min = divmod(min, 60) 
    return "%d:%02d:%02d" % (hour, min, sec) 