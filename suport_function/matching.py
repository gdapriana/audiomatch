import pandas as pd
import os
from core_functions.dtw.compare import compare
# {
#   'matching_name': 'train_10_with_test_normal_50_10',
#   'train_path': 'resources/features/train/1_0',
#   'test_path': 'resources/features/test/normal/50/1_0',
#   'train_csv': 'resources/csv/train.csv',
#   'test_csv': 'resources/csv/test_normal_50.csv',
#   'out_path': 'resources/matching/train_10_with_test_normal_50_10.csv'
# },
def matching_features(matching_data) -> None:
  for matching in matching_data:
    matching_name = matching['matching_name']
    train_npy_path = matching['train_path']
    test_npy_path = matching['test_path']
    train_csv = pd.read_csv(matching['train_csv'])
    test_csv = pd.read_csv(matching['test_csv'])
    out_path = matching['out_path']

    if os.path.exists(out_path):
      print(f"{matching_name} already compared!")
      continue

    all_train_npy = []
    all_test_npy = []

    for i, row in train_csv.iterrows():
      all_train_npy.append({ 'title': row['title'], 'artist': row['artist'], 'npy_path': f"{train_npy_path}/{row['title']}.npy" })

    for i, row in test_csv.iterrows():
      all_test_npy.append({ 'title': row['title'], 'artist': row['artist'], 'npy_path': f"{test_npy_path}/{row['title']}.npy" })

    print(f"comparing: { matching_name }...")
    result = compare(all_train_npy, all_test_npy)
    pd.DataFrame(result).to_csv(out_path, index=False)
    print("done!\n")

