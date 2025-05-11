import numpy as np
from typing import Dict, Union
default_params: Dict[str, Union[Dict[str, int | float], int]] = {
  "emphasis": {"coefficients": 0.97},
  "framing": {"frame_size": 1, "frame_hop": 0.5},
  "fft": {"n_fft": 512},
  "melbank": {"filter": 40},
  "dct": { 'n_mfcc': 13 },
}

def pre_emphasis(
  signal: np.ndarray = None,
  coefficients: float = default_params['emphasis']['coefficients']
) -> np.ndarray:
  
  # tahap 1: Input (signal dan coefficients)
  
  # validasi parameter
  if signal is None: raise TypeError("signal cannot be empty")
  if not isinstance(signal, np.ndarray): raise TypeError("signal should be numpy array")

  # tahap 2: proses utama
  return np.append(signal[0], signal[1:] - coefficients * signal[:-1])
  # tahap 3: output
  
def frame_blocking(
  signal: np.ndarray = None,
  sampling_rate: int = None,
  frame_size: float = default_params['framing']['frame_size'],
  frame_hop: float = default_params['framing']['frame_hop']
) -> np.ndarray:
  # tahap 1: input (signal, sampling_rate, frame_size, frame_hop).

  # validasi parameter.
  if signal is None: raise TypeError("signal cannot be empty")
  if sampling_rate is None: raise TypeError("sampling_rate cannot be empty")
  if not isinstance(signal, np.ndarray): raise TypeError("signal should be numpy array")
  if not isinstance(sampling_rate, int): raise TypeError("sampling_rate should be float")

  # tahap 2: Hitung panjang dan langkah frame dalam sampel.
  frame_length = np.round(frame_size * sampling_rate).astype(int)
  frame_step = np.round(frame_hop * sampling_rate).astype(int)
  signal_length = signal.shape[0]

  # tahap 3: Hitung jumlah frame yang diperlukan.
  n_frames = np.ceil(abs(signal_length - frame_length) / frame_step).astype(int)

  # tahap 4: Padding sinyal (mengisi 0 untuk menyamaratakan size frame).
  pad_signal_length = int(n_frames * frame_step + frame_length)
  zeros_pad = np.zeros((1, pad_signal_length - signal_length))
  pad_signal = np.concatenate((signal.reshape((1, -1)), zeros_pad), axis=1).reshape(-1)

  # tahap 5: Inisialisasi array untuk frame dan indeks.
  frames = np.zeros((n_frames, frame_length))
  indices = np.arange(0, frame_length)

  # tahap 6: Looping untuk memotong frame.
  for i in np.arange(0, n_frames):
    offset = i * frame_step
    frames[i] = pad_signal[(indices + offset)]

  # tahap 7: Output.
  return frames

def windowing(
  signal: np.ndarray = None
) -> np.ndarray:
  # tahap 1: Input (signal).

  # validasi parameter.
  if signal is None: raise TypeError("signal cannot be empty")
  if not isinstance(signal, np.ndarray): raise TypeError("signal should be numpy array")

  # tahap 2: Hitung panjang window.
  window_length = signal.shape[1]
  n = np.arange(0, window_length)

  # tahap 3: Buat Hamming window.
  h = 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_length - 1))

  # tahap 4: Aplikasikan window ke sinyal.
  signal *= h

  # tahap 5: Output.
  return signal