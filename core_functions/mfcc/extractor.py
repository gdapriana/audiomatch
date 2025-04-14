import librosa
import numpy as np
from typing import Union

from core_functions.mfcc.prepocessing import pre_emphasis, frame_blocking, windowing
from core_functions.mfcc.mfcc import fft, mel_filterbank, signal_power_to_db, cosine_transform

def extract_mfcc(
  audio_path: str,
  emphasis: float,
  frame_size: float,
  frame_hop: float,
  n_fft: int,
  n_mels: int,
  n_mfcc: int,
  audio_title: str = None,
  out_path: str = None,
) -> Union[np.ndarray, None]:
  """
  Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from an audio file.

  This function performs a series of preprocessing steps, including pre-emphasis,
  framing, windowing, FFT, Mel filterbank application, logarithmic scaling,
  and discrete cosine transform (DCT), to compute the MFCC features.

  :param audio_title: A string representing the title of the audio file.
  :param audio_path: A string representing the path to the input audio file.
  :param out_path: A string representing the directory path to save the extracted MFCC features (if not None).
  :param emphasis: A float representing the pre-emphasis filter coefficient.
  :param frame_size: A float representing the frame size in seconds.
  :param frame_hop: A float representing the frame hop (stride) in seconds.
  :param n_fft: An integer representing the number of FFT points.
  :param n_mels: An integer representing the number of Mel filter banks.
  :param n_mfcc: An integer representing the number of MFCC coefficients to retain.

  :return:
      - A 2D NumPy array of shape `(n_frames, n_mfcc)`, containing the extracted MFCC features if `out_path` is None.
      - `None` if the features are saved to a file.

  :raises ValueError: If any MFCC parameter (`emphasis`, `frame_size`, `frame_hop`, `n_fft`, `n_mels`, `n_mfcc`) is None.
  :raises FileNotFoundError: If the specified `audio_path` does not exist.

  :example:
      ```python
      mfcc_features = extract_mfcc(
          audio_title="example",
          audio_path="audio/example.wav",
          out_path=None,
          emphasis=0.97,
          frame_size=1,
          frame_hop=0.5,
          n_fft=512,
          n_mels=40,
          n_mfcc=13
      )
      ```
  """

  # params validation
  if emphasis is None or frame_size is None or frame_hop is None or n_fft is None or n_mels is None or n_mfcc is None:
    raise ValueError('mfcc parameters cannot be none')

  if audio_title is not None:
    print(f"extracting {audio_title}...")

  signal, sampling_rate = librosa.load(audio_path)
  signal = 1.0 * signal

  emphased_signal = pre_emphasis(signal=signal, coefficients=emphasis)
  framed_signal = frame_blocking(signal=emphased_signal, sampling_rate=sampling_rate, frame_size=frame_size, frame_hop=frame_hop)
  windowed_signal = windowing(signal=framed_signal)
  fft_signal = fft(signal=windowed_signal, n_fft=n_fft)
  mel_signal, _ = mel_filterbank(signal=fft_signal, sampling_rate=sampling_rate , n_mels=n_mels, f_max=sampling_rate / 2, f_min=0)
  log_spec_frames = signal_power_to_db(signal=mel_signal)
  features = cosine_transform(signal=log_spec_frames, coefficients=n_mfcc)

  if out_path is not None:
    np.save(f"{out_path}/{audio_title}.npy", features)
    return None
  else:
    return features