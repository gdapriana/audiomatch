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
  """
  :param signal: A 1D NumPy array representing the input audio signal.
  :param coefficients: A float value representing the pre-emphasis filter coefficient (default taken from `default_params`).
  :return: A 1D NumPy array containing the filtered signal after applying pre-emphasis.
  """

  # params validation
  if signal is None: raise TypeError("signal cannot be empty")
  if not isinstance(signal, np.ndarray): raise TypeError("signal should be numpy array")

  return np.append(signal[0], signal[1:] - coefficients * signal[:-1])

def frame_blocking(
  signal: np.ndarray = None,
  sampling_rate: int = None,
  frame_size: float = default_params['framing']['frame_size'],
  frame_hop: float = default_params['framing']['frame_hop']
) -> np.ndarray:

  """
    :param signal: A 1D NumPy array representing the input audio signal.
    :param sampling_rate: A float representing the sampling rate of the audio signal (in Hz).
    :param frame_size: A float representing the frame duration in seconds (default taken from `default_params`).
    :param frame_hop: A float representing the hop duration between consecutive frames in seconds (default from `default_params`).
    :return: A 2D NumPy array of shape `(n_frames, frame_length)`, where each row is a frame of the signal.
  """

  # params validation
  if signal is None: raise TypeError("signal cannot be empty")
  if sampling_rate is None: raise TypeError("sampling_rate cannot be empty")
  if not isinstance(signal, np.ndarray): raise TypeError("signal should be numpy array")
  if not isinstance(sampling_rate, int): raise TypeError("sampling_rate should be float")

  # init variables
  frame_length = np.round(frame_size * sampling_rate).astype(int)
  frame_step = np.round(frame_hop * sampling_rate).astype(int)
  signal_length = signal.shape[0]

  # calculate number of frames
  n_frames = np.ceil(abs(signal_length - frame_length) / frame_step).astype(int)

  # pad signal that all frames have equal number of samples
  pad_signal_length = int(n_frames * frame_step + frame_length)
  zeros_pad = np.zeros((1, pad_signal_length - signal_length))
  pad_signal = np.concatenate((signal.reshape((1, -1)), zeros_pad), axis=1).reshape(-1)

  # extract frames
  frames = np.zeros((n_frames, frame_length))
  indices = np.arange(0, frame_length)

  for i in np.arange(0, n_frames):
    offset = i * frame_step
    frames[i] = pad_signal[(indices + offset)]

  return frames

def windowing(
  signal: np.ndarray = None
) -> np.ndarray:
  """
  Applies a Hamming window to each frame of the input signal.
  The Hamming window helps reduce spectral leakage by smoothly tapering the edges of each frame.
  It is commonly used in signal processing before applying FFT.
  :param signal: A 2D NumPy array of shape `(n_frames, frame_length)`, where each row represents a frame of the signal.
  :return: A 2D NumPy array of the same shape as `signal`, with the Hamming window applied to each frame.
  :raises TypeError: If `signal` is None.
  :raises TypeError: If `signal` is not a NumPy array.
  """

  # params validation
  if signal is None: raise TypeError("signal cannot be empty")
  if not isinstance(signal, np.ndarray): raise TypeError("signal should be numpy array")

  window_length = signal.shape[1]
  n = np.arange(0, window_length)

  # set window coefficients
  h = 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_length - 1))

  # perform windowing
  signal *= h

  return signal