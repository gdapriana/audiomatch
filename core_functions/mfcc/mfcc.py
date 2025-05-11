import numpy as np
from scipy.fftpack import dct
from typing import Dict, Union

default_params: Dict[str, Union[Dict[str, int | float], int]] = {
  "emphasis": {"coefficients": 0.97},
  "framing": {"frame_size": 1, "frame_hop": 0.5},
  "fft": {"n_fft": 512},
  "melbank": {"filter": 40},
  "dct": { 'n_mfcc': 13 },
}

def fft(
  signal: np.ndarray = None,
  n_fft: int = default_params['fft']['n_fft']
):
  
  # tahap 1: Input (signal, n_fft).

  # validasi parameter
  if signal is None: raise TypeError("signal cannot be empty")
  if not isinstance(signal, np.ndarray): raise TypeError("signal should be numpy array")

  # untuk mempermudah dalam koding, fft diambil dari library numpy.
  # tahap 2 sampai dengan tahap 7
  spec_frames = np.fft.rfft(signal, n=n_fft, axis=1)
  mag_spec_frames = np.abs(spec_frames)
  pow_spec_frames = (mag_spec_frames ** 2) / mag_spec_frames.shape[1]

  # tahap 8: output
  return pow_spec_frames

def mel_filterbank(
  f_max,
  f_min,
  signal: np.ndarray = None,
  n_mels: int = default_params['melbank']['filter'],
  sampling_rate: int = None,
):
  # tahap 1: Input (f_max, f_min, signal, n_mels, sampling_rate)

  # tahap 2: Hitung ukuran fft.
  n_fft = signal.shape[1] - 1
  
  # tahap 3: Konversi Frekuensi ke Skala Mel.
  mel_lf = 2595 * np.log10(1 + f_min / 700)
  mel_hf = 2595 * np.log10(1 + f_max / 700)
  mel_points = np.linspace(mel_lf, mel_hf, n_mels + 2)

  # tahap 4: Konversi Titik Mel ke Frekuensi Hz.
  hz_points = 700 * (np.power(10, mel_points / 2595) - 1)

  # tahap 5: Konversi ke Bin FFT (indeks).
  fft_bank_bin = np.floor((n_fft + 1) * hz_points / (sampling_rate / 2))
  fft_bank_bin[-1] = n_fft

  # tahap 6:  Inisialisasi Matriks Filterbank.
  f_bank = np.zeros((n_mels, n_fft + 1))
  
  # tahap 7: Looping (perulangan) untuk membangun filter segitiga.
  for i in np.arange(1, n_mels + 1):
    left_f = int(fft_bank_bin[i - 1])
    center_f = int(fft_bank_bin[i])
    right_f = int(fft_bank_bin[i + 1])

    for k in np.arange(left_f, center_f + 1):
      f_bank[i - 1, k] = (k - left_f) / (center_f - left_f)

    for k in np.arange(center_f, right_f + 1):
      f_bank[i - 1, k] = (-k + right_f) / (-center_f + right_f)

    # scale filter bank by its width
    f_bank[i - 1] /= (hz_points[i] - hz_points[i-1])

  # tahap 8: Terapkan Filter ke Spektrum.
  filtered_frames = np.dot(signal, f_bank.T)
  filtered_frames += np.finfo(float).eps
  
  # tahap 9: Output:
  return filtered_frames, hz_points


def signal_power_to_db(
  signal: np.ndarray,
  min_amp=1e-10, top_db=80
) -> np.ndarray:
  """
  Covert power spectrum amplitude to dB.
  :param signal: Power spectrum frames
  :param min_amp: Minimum amplitude
  :param top_db: Max value ib dB
  :return: Power dB frames
  """
  log_spec = 10.0 * np.log10(np.maximum(min_amp, signal))
  log_spec = np.maximum(log_spec, log_spec.max() - top_db)

  return log_spec

def cosine_transform(
  signal: np.array,
  coefficients: float = default_params['dct']['n_mfcc']
) -> np.ndarray:
  # tahap 1: Input (signal, coefficients)
  
  # untuk mempermudah dalam koding, dct diambil dari library scipy.
  # tahap 2 sampai tahap 4, (pada tahap 5: Pilih koefisien MFCC dari hasil DCT dengan jumlah yang telah ditentukan)
  dct_signal = dct(x=signal)[:, 1 : (coefficients + 1)]
  
  # tahap 6: Output
  return dct_signal