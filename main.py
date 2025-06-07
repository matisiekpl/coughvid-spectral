import os.path

import pandas

from nn import train
from process import aggregate_features, load_dataset, load_features


def create_dataset():
    if os.path.exists('dataset.csv'):
        print('Skipping feature extraction, because dataset.csv exists')
        return pandas.read_csv('dataset.csv')

    processed_df = load_dataset()
    print(f'Extracting spectrogram features from {len(processed_df)} waveforms')
    uuid, x = load_features(processed_df)
    dataset = aggregate_features(processed_df, uuid, x)
    print(f'Dataset records count: {len(dataset)}')
    dataset.to_csv('dataset.csv', index=False)
    return dataset


if __name__ == '__main__':
    data = create_dataset()
    train(data)
