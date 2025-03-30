from typing import Dict, Union

default_params: Dict[str, Union[Dict[str, int | float], int]] = {
  "emphasis": {"coefficients": 0.97},
  "framing": {"frame_size": 1, "frame_hop": 0.5},
  "fft": {"n_fft": 512},
  "melbank": {"filter": 40},
  "dct": { 'n_mfcc': 13 },
}
