import streamlit as st
from suport_function.extract import extract_single
import pandas as pd

from suport_function.matching import matching_single

params: dict = {
  "emphasis": 0.97,
  "frame_size": 1,
  "frame_hop": 0.5,
  "n_fft": 512,
  "n_mels": 40,
  "n_mfcc": 13
}

if __name__ == '__main__':
  audio = st.file_uploader("Upload an audio file", type=["mp3"])
  train_csv = pd.read_csv("resources/csv/train.csv")
  database = []

  for i, row in train_csv.iterrows():
    database.append({'title': row['title'], 'npy_path': f"resources/features/train/1_0/{row['title']}.npy"})

  if audio is not None:
    features = extract_single(audio, params)
    result = matching_single(database, features)
    st.text(result)
