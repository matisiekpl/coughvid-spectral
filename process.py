import json
import os

import librosa
import numpy as np
import pandas as pd
import kagglehub
from tqdm import tqdm

ROOT = kagglehub.dataset_download("nasrulhakim86/coughvid-wav") + '/public_dataset/'
audio_length = 22050


def load_dataset(take=None):
    print(f'Using dataset from {ROOT}')
    json_files = []
    for file in os.listdir(ROOT):
        if file.endswith('.json'):
            with open(os.path.join(ROOT, file)) as f:
                json_data = json.load(f)
                json_data['uuid'] = file.split('.')[0]
                json_files.append(json_data)

    df = pd.DataFrame(json_files)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    if take is not None:
        df = df.head(take)

    df['cough_detected'] = df['cough_detected'].astype(float)

    # drop null status
    df = df[df.status.notna()]
    # drop cough_detected < 0.8
    df = df[df.cough_detected >= 0.8]
    return df


def segment_cough(x, fs, cough_padding=0.2, min_cough_len=0.2, th_l_multiplier=0.1, th_h_multiplier=2):
    # Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power
    cough_mask = np.array([False] * len(x))

    # Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    cough_segments = []
    padding = round(fs * cough_padding)
    min_cough_samples = round(fs * min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01 * fs)
    below_th_counter = 0

    for i, sample in enumerate(x ** 2):
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i + padding if (i + padding < len(x)) else len(x) - 1
                    cough_in_progress = False
                    if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
                        cough_segments.append(x[cough_start:cough_end + 1])
                        cough_mask[cough_start:cough_end + 1] = True
            elif i == (len(x) - 1):
                cough_end = i
                cough_in_progress = False
                if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
                    cough_segments.append(x[cough_start:cough_end + 1])
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = i - padding if (i - padding >= 0) else 0
                cough_in_progress = True

    return cough_segments, cough_mask


def extract_features(audio_data, sample_rate):
    features = []
    stft = np.abs(librosa.stft(audio_data))

    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    features.extend(mfcc)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    features.extend(chroma)

    mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
    features.extend(mel)

    min_val = 0.5 * sample_rate * 2 ** (-6)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate, fmin=min_val).T, axis=0)
    features.extend(contrast)

    return np.array(features)


def load_features(df):
    features, filenames = [], []
    for idx in tqdm(range(len(df))):
        filename = df.uuid.iloc[idx]
        path = os.path.join(ROOT + filename + '.wav')

        audio, sample_rate = librosa.load(path, mono=True)
        # Segment each audio into individual coughs using a hysteresis comparator on the signal power
        cough_segments, cough_mask = segment_cough(audio, sample_rate, min_cough_len=0.1, cough_padding=0.1,
                                                   th_l_multiplier=0.1, th_h_multiplier=2)

        if len(cough_segments) > 0:
            i = 0
            for audio in cough_segments:
                i += 1
                if len(audio) > 8000:
                    if len(audio) < audio_length:
                        audio_pad = librosa.util.pad_center(audio, size=audio_length)
                    else:
                        audio_pad = audio[:audio_length]

                feature = extract_features(audio_pad, sample_rate)
                features.append(feature)
                filenames.append(filename)

    return np.array(filenames), np.array(features)


def aggregate_features(processed_df, uuid, X):
    x_mfcc = X[:, 0:40]
    x_chroma = X[:, 40:52]
    x_mel = X[:, 52:180]
    x_contrast = X[:, 180:]

    uuid_df = pd.DataFrame({'uuid': uuid})
    mfcc_df = pd.DataFrame(x_mfcc)
    mfcc_df.columns = ["mfcc" + str(i) for i in range(1, x_mfcc.shape[1] + 1)]

    mel_df = pd.DataFrame(x_mel)
    mel_df.columns = ["mel" + str(i) for i in range(1, x_mel.shape[1] + 1)]

    chroma_df = pd.DataFrame(x_chroma)
    chroma_df.columns = ["chr" + str(i) for i in range(1, x_chroma.shape[1] + 1)]

    contrast_df = pd.DataFrame(x_contrast)
    contrast_df.columns = ["con" + str(i) for i in range(1, x_contrast.shape[1] + 1)]

    all_df = pd.concat([uuid_df, mfcc_df, mel_df, chroma_df, contrast_df], axis=1)
    all_df.head(3)

    label_df = processed_df[['uuid', 'status']].reset_index(drop=True)

    dataset = pd.merge(all_df, label_df, on='uuid')
    dataset = dataset[dataset.status != 'unknown']
    dataset = dataset.groupby('status').sample(n=2185)

    return dataset
