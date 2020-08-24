# Behind every great piece of software, is another separate file, full of shit
#
import os
import glob
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
import shutil
from python_speech_features import mfcc
import math
import time

# item is a file you want to use/make
def make_folder(item):
    if not os.path.exists(item):
        print(item + ' Does Not Exist, Creating')
        os.mkdir(item)


def convert_mp3_to_wav(host_dir, target_dir):
    mp3 = host_dir
    wav = target_dir

    for item in [mp3, wav]:
        if not os.path.exists(item):
            print(item + ' Does Not Exist, Creating')
            os.mkdir(item)

    mp3_files = glob.glob(mp3 + '*.mp3')
    wav_files = glob.glob(wav + '*.wav')

    mp3_names = []
    for mp3_file in mp3_files:
        mp3_name = os.path.splitext(os.path.basename(mp3_file))
        mp3_names.append(mp3_name[0])

    wav_names = []
    for wav_file in wav_files:
        wav_name = os.path.splitext(os.path.basename(wav_file))
        wav_names.append(wav_name[0])

    # finds which files in the mp3 folder are not in the wav folder.
    list_difference = [name for name in mp3_names if name not in wav_names]

    for name in list_difference:
        print(name + ' Began')
        mp3_name = os.path.join(mp3, name + ".mp3")
        sound = AudioSegment.from_mp3(mp3_name)
        wav_name = os.path.join(wav, name + ".wav")
        sound.export(wav_name, format="wav")
        print(name + ' done')

# Checks if wav files can be read and appends them to a list
# path = path to a folder containing wav files you want to append to a list
def process_wav_or_mp3_files(path):

    convert_mp3_to_wav(path, path)

    list_ = []
    if os.path.exists(path):
        for filename in glob.glob(os.path.join(path, '*.wav')):
            list_.append(filename)
    else:
        print(path + ' does not exist')

    list2_ = []
    exceptions = 0
    for file in list_:
        try:
            fs, data = wavfile.read(file)  # read
            data = np.array(data, dtype=np.float32)  # this stops the process from breaking if a file cant be rea
            list2_.append(file)
        except:
            exceptions += 1
    print('no of exceptions: ' + str(exceptions))
    return list2_


# A function to copy files from one place to another
def copy_files_to_folder(array, destination):
    print(destination)
    # delete all files currently in the destination folder
    if not os.path.exists(destination):
        os.makedirs(destination)

    for the_file in os.listdir(destination):
        file_path = os.path.join(destination, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)

        except Exception as e:
            print(e)
    # replace them with files in array
    for f in array:
        shutil.copy(f, destination)
    print('done')


# Audio Processing
def make_mono(data):
    try:
        if len(data[0]) == 2:
            data = np.array(data).T
            data = normalise(data[0] + data[1])
    except:
        data = data
    return data


def normalise(x):
    x = x / max(x)
    return x


def normalise_stereo(wav):
    temp = np.array(wav).T
    max_t = max([max(temp[0]), max(temp[1])])
    return wav / max_t


def half_samplerate(fs, data, target_samplerate):
    if target_samplerate == fs / 2:
        output = []
        for i in range(len(data)):
            if i % 2 == 0:
                output.append(data[i])
        output_samplerate = target_samplerate
    elif target_samplerate == fs:
        output = data
        output_samplerate = fs
    else:
        output_samplerate = fs
    return output_samplerate, output


def remove_padding(live_set):
    offset = 0
    for i in live_set:
        if i[0] == 0:
            offset += 1
        if i[0] != 0:
            break

    return live_set[offset:]


# Takes in a signal and makes it 22.05kHz and Mono
def preprocess(fs, sig):
    # frame pre-processing
    # 1. if fs = 44,100 then half sample-rate
    if fs == 44100:
        fs, sig = half_samplerate(fs, sig, 22050)
    if fs == 22050:
        sig = make_mono(sig)
        sig = normalise(sig)
        return fs, sig
    else:
        print('samplerate must be 44.1kHz or 22.05kHz')


# Percepts
def zero_cross_rate(fs, N, block):
    zero_crosses = np.nonzero(np.diff(block > 0))
    no_crosses = np.size(zero_crosses) * 0.5
    cross_rate = no_crosses * fs / N
    return cross_rate


# RMS - oscillating signal level
def rms_value(block):
    rms = np.sqrt(np.mean(block ** 2))
    return rms


# Spectral Centroid - where the centre of spectral spread lies
def spectral_centroid(samplerate, block):
    magnitudes = np.abs(np.fft.rfft(block))  # magnitudes of positive frequencies
    length = len(block)
    freqs = np.abs(np.fft.fftfreq(length, 1.0 / samplerate)[:length // 2 + 1])  # positive frequencies
    return np.sum(magnitudes * freqs) / np.sum(magnitudes)  # return weighted mean


# Spectral complexity - Max=White noise, Min=Pure sine tone
def spectral_entropy(block):
    # this function does not handle '0' 
    # looks like i should definitely normalise this to how long the input array is
    block_fft = np.fft.fft(block)
    p_w = (1 / len(block_fft)) * abs(block_fft) ** 2
    try:
        p_i = p_w / sum(p_w)
    except:
        p_i = 0
        print('set to 0 because of 0 division')
    pse = 0
    for p in p_i:
        pse += p * np.log(p)
    spec_entropy = -pse
    if math.isnan(spec_entropy):
        spec_entropy = 0
    
    return spec_entropy


def average(list_in):
    return sum(list_in) / len(list_in)


def percepts(fs, frame, window_length=0.04):

    nfft_in = 0
    size = True
    x = 0

    n_windows_in_signal = int(math.floor(len(frame) / (fs * window_length)))

    zcr = []
    spec_cent = []
    spec_ent = []
    mfcc_values = []
    rms = []
    for n_window in range(n_windows_in_signal):
        a = int(n_window * fs * window_length)
        b = int((n_window + 1) * fs * window_length)
        window = frame[a:b] * np.hanning(len(frame[a:b]))

        ## Extract features from window
        zcr.append(zero_cross_rate(fs, len(window), window))
        spec_cent.append(spectral_centroid(fs, window))
        spec_ent.append(spectral_entropy(window))
        rms.append(rms_value(window))

        while size == True:
            if len(window) > nfft_in:
                nfft_in = 2 ** x
            x += 1
            if x == 13 or nfft_in > len(window):
                break

        mfcc_val = mfcc(window, fs, winlen=window_length, nfft=nfft_in, winstep=window_length)
        mfcc_values.append(mfcc_val[0])

        # simplest thing to do is to take the mean average of each of these percepts over the 1s period
    ave_zcr = average(zcr)
    ave_spec_cent = average(spec_cent)
    ave_spec_ent = average(spec_ent)
    ave_rms = average(rms)

    mfcc_values = np.array(mfcc_values)
    ave_mfcc_values = average(mfcc_values)

    percept_dict = {
        'ave_zcr': ave_zcr,
        'zcr': zcr,
        'ave_spec_cent': ave_spec_cent,
        'spec_cent': spec_cent,
        'ave_spec_ent': ave_spec_ent,
        'spec_ent': spec_ent,
        'ave_rms': ave_rms,
        'rms': rms,
        'ave_mfcc': ave_mfcc_values,
        'mfcc': mfcc_values,
        'frame_length': len(frame) / fs
    }
    return percept_dict


def sec_to_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# Main Functions
def get_classification(input_signal, fs, classifier, rms_threshold=0.09, verbose=False, frame_length=0.5, tic=0):
    fs, sig = preprocess(fs, input_signal)
    n_frames_in_signal = int(math.floor(len(sig) / (fs * frame_length)))
    pred_bool = True

    predictions = []
    log = []

    if verbose:
        print('Processing')
        print('Frames:' + str(n_frames_in_signal))

 
    rms = rms_value(input_signal)
    if rms < rms_threshold:   

        # -- 
        for n_frame in range(n_frames_in_signal):

            # sometimes things take a while so....
            if verbose:
                if n_frame % 100 == 0:
                    print(str(n_frame) + '/' + str(n_frames_in_signal))
                    c_time = time.perf_counter()
                    time_taken_s = c_time - tic
                    print('Time Taken: ' + sec_to_time(time_taken_s))

            a = int(n_frame * fs * frame_length)
            b = int((n_frame + 1) * fs * frame_length)
            frame = sig[a:b]

            out = percepts(fs, frame)
            # Order in is [mfcc, spec_cent, spec_ent, zcr]

            x = []
            for value in out['mfcc']:
                x.extend(value)
            x.extend(out['ave_mfcc'])
            x.extend([out['ave_spec_cent']])
            x.extend(out['spec_cent'])
            x.extend([out['ave_spec_ent']])
            x.extend(out['spec_ent'])
            x.extend([out['ave_zcr']])
            x.extend(out['zcr'])

            x = np.array(x)
            x = x.reshape(1, -1)

            # temporary 'isnan' detection - basically just handles 0 errors after the fact
            # this CANNOT be included in training and testing!!!!!!!!!! it would cause false representaiton 
            for i in range(len(x[0])):
                if math.isnan(x[0][i]):
                    x[0][i] = 0 # replace nan's with 0's 

            pred = classifier.predict(x)

            predictions.append(pred)
            log.append(a)
        # -- 
    if rms < rms_threshold:  
        log_fs = 1 / ((log[2] - log[1]) / fs)
        predictions = np.array(predictions).T[0]
    else:
        log_fs = False
        predictions = False
        pred_bool = False

    return predictions, log_fs, pred_bool


def make_decision(predictions, log_fs, pred_bool, n_noise_thresh=5):
    cut = False
    cut_point = False

    if pred_bool:
        i = 0
        count = 0
        for pred in predictions:
            if pred == 1:
                count += 1
                first_noise = i
            if count == n_noise_thresh:
                cut_point = first_noise / log_fs
                cut = True
                break
            i += 1

    return cut, cut_point
