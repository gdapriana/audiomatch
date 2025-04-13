import numpy as np
import pandas as pd
import os
from core_functions.mfcc.extractor import extract_mfcc

def extract_train(train_data)->None:
  for case in train_data:
    if os.path.exists(case['out_path']) and os.listdir(case['out_path']):
      print(f"{case['out_path']} already extracted")
      continue

    print(f"\nextracting train {case['out_path']}...")
    csv = pd.read_csv(case['csv_path'])
    for i, row in csv.iterrows():
      song_path = f"{case['audio_path']}/{row['title']}.mp3"
      extract_mfcc(
        audio_title=row['title'],
        audio_path=song_path,
        emphasis=case['params']['emphasis'],
        frame_size=case['params']['frame_size'],
        frame_hop=case['params']['frame_hop'],
        n_fft=case['params']['n_fft'],
        n_mels=case['params']['n_mels'],
        n_mfcc=case['params']['n_mfcc'],
        out_path=case['out_path']
      )


def extract_test(test_data) -> None:
    for duration, duration_value in test_data.items():
      for case in duration_value:
        if os.path.exists(case['out_path']) and os.listdir(case['out_path']):
          print(f"{case['out_path']} already extracted")
          continue
        print(f"\nextracting test {case['out_path']}...")
        csv = pd.read_csv(case['csv_path'])
        for i, row in csv.iterrows():
          song_path = f"{case['audio_path']}/{row['title']}.mp3"
          extract_mfcc(
            audio_title=row['title'],
            audio_path=song_path,
            emphasis=case['params']['emphasis'],
            frame_size=case['params']['frame_size'],
            frame_hop=case['params']['frame_hop'],
            n_fft=case['params']['n_fft'],
            n_mels=case['params']['n_mels'],
            n_mfcc=case['params']['n_mfcc'],
            out_path=case['out_path']
          )

def extract_single(audio, params: dict) -> np.ndarray:
  return extract_mfcc(
    audio_path=audio,
    emphasis=params['emphasis'],
    frame_size=params['frame_size'],
    frame_hop=params['frame_hop'],
    n_fft=params['n_fft'],
    n_mels=params['n_mels'],
    n_mfcc=params['n_mfcc']
  )