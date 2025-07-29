import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, entropy, skew
import nolds
from scipy.signal import welch, find_peaks
from joblib import dump

def extract_features_from_signal(signal, sampling_rate=1024):
    """
    Extract statistical and frequency-domain features from a 1D vibration signal.
    """
    signal = np.array(signal)
    features = {}

    features['Kurtosis'] = kurtosis(signal)
    ent = entropy(signal)
    features['Entropy'] = 0 if ent < -1e-6 else ent
    features['Fractal Dimension'] = nolds.dfa(signal)
    features['Peak factor'] = np.max(np.abs(signal)) / np.sqrt(np.mean(np.square(signal)))
    features['Pulse factor'] = np.max(np.abs(signal)) / np.mean(np.abs(signal))
    features['Crest factor'] = np.max(np.abs(signal)) / np.mean(np.sqrt(np.mean(np.square(signal))))

    freqs, power = welch(signal, fs=sampling_rate)
    peak_freqs, _ = find_peaks(power, height=np.mean(power))
    total_energy = np.sum(power)
    peak_energy = np.sum(power[peak_freqs])
    features['Energy ratio'] = peak_energy / total_energy
    features['Spectral flatness'] = np.exp(np.mean(np.log(power))) / np.mean(power)

    features['Mean'] = np.mean(signal)
    features['Variance'] = np.var(signal)
    features['Skewness'] = skew(signal)
    features['Peak vibration'] = np.max(np.abs(signal))
    features['Rms vibration'] = np.sqrt(np.mean(np.square(signal)))

    return features

def get_bearing_data(folder):
    """
    Load bearing vibration CSV files from the given folder and extract features.
    Returns:
        - all_data: np.ndarray of shape [N, 2] (horizontal and vertical)
        - features_df: DataFrame with per-file features
    """
    files = sorted([f for f in os.listdir(folder) if 'acc' in f and f.endswith('.csv')])
    file_paths = [os.path.join(folder, f) for f in files]

    # Detect CSV separator
    sep = ';' if pd.read_csv(file_paths[0], header=None).shape[1] == 1 else ','

    h_signals, v_signals, features = [], [], []

    for f in file_paths:
        data = pd.read_csv(f, header=None, sep=sep)
        h = data.iloc[:, -2].values
        v = data.iloc[:, -1].values
        h_signals.append(h)
        v_signals.append(v)
        features.append(extract_features_from_signal(h))

    all_data = np.stack([np.concatenate(h_signals), np.concatenate(v_signals)], axis=-1)
    return all_data, pd.DataFrame(features)

def compute_rul(features_df):
    """
    Add normalized RUL to the features dataframe.
    """
    total = len(features_df)
    features_df['rul'] = [(total - i) / total for i in range(1, total + 1)]
    return features_df

def save_processed_data(all_data, features_df, prefix, out_dir='./processed'):
    """
    Save processed raw signal and features using joblib and CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    dump(all_data, os.path.join(out_dir, f'{prefix}_all_data'))
    dump(features_df, os.path.join(out_dir, f'{prefix}_features_df'))
    features_df.to_csv(os.path.join(out_dir, f'samples_data_{prefix}.csv'), index=False)
    print(f'[INFO] Saved files for {prefix} to {out_dir}')

if __name__ == '__main__':
    # Example usage
    bearing_path = './phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1'
    prefix = 'Bearing1_1'

    all_data, features_df = get_bearing_data(bearing_path)
    features_df = compute_rul(features_df)
    save_processed_data(all_data, features_df, prefix)
