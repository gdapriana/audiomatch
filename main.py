from core_functions.evaluation.evaluation import evaluation
from suport_function.extract import extract_train, extract_test
from suport_function.matching import matching_features

train_data = [
  {
    'csv_path': 'resources/csv/train.csv',
    'params': {'frame_size': 0.2, 'frame_hop': 0.1, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
    'audio_path': 'resources/dataset/train',
    'out_path': 'resources/features/train/0_2'
  },
  {
    'csv_path': 'resources/csv/train.csv',
    'params': {'frame_size': 0.5, 'frame_hop': 0.25, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
    'audio_path': 'resources/dataset/train',
    'out_path': 'resources/features/train/0_5'
  },
  {
    'csv_path': 'resources/csv/train.csv',
    'params': {'frame_size': 1, 'frame_hop': 0.5, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
    'audio_path': 'resources/dataset/train',
    'out_path': 'resources/features/train/1_0'
  },
  {
    'csv_path': 'resources/csv/train.csv',
    'params': {'frame_size': 1.5, 'frame_hop': 0.75, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
    'audio_path': 'resources/dataset/train',
    'out_path': 'resources/features/train/1_5'
  },
]

test_data = {
    'duration_30': [
      {
        'csv_path': 'resources/csv/train.csv',
        'params': {'frame_size': 0.2, 'frame_hop': 0.1, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/30',
        'out_path': 'resources/features/test/30/0_2'
      },
      {
        'csv_path': 'resources/csv/train.csv',
        'params': {'frame_size': 0.5, 'frame_hop': 0.25, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/30',
        'out_path': 'resources/features/test/30/0_5'
      },
      {
        'csv_path': 'resources/csv/train.csv',
        'params': {'frame_size': 1.0, 'frame_hop': 0.5, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/30',
        'out_path': 'resources/features/test/30/1_0'
      },
      {
        'csv_path': 'resources/csv/train.csv',
        'params': {'frame_size': 1.5, 'frame_hop': 0.75, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/30',
        'out_path': 'resources/features/test/30/1_5'
      },
    ],
    'duration_50': [
      {
        'csv_path': 'resources/csv/train.csv',
        'params': {'frame_size': 0.2, 'frame_hop': 0.1, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/50',
        'out_path': 'resources/features/test/50/0_2'
      },
      {
        'csv_path': 'resources/csv/train.csv',
        'params': {'frame_size': 0.5, 'frame_hop': 0.25, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/50',
        'out_path': 'resources/features/test/50/0_5'
      },
      {
        'csv_path': 'resources/csv/train.csv',
        'params': {'frame_size': 1.0, 'frame_hop': 0.5, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/50',
        'out_path': 'resources/features/test/50/1_0'
      },
      {
        'csv_path': 'resources/csv/train.csv',
        'params': {'frame_size': 1.5, 'frame_hop': 0.75, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/50',
        'out_path': 'resources/features/test/50/1_5'
      },
    ],
}

matching_data = [
  {
    'matching_name': 'train_02_with_test_30_02',
    'train_path': 'resources/features/train/0_2',
    'test_path': 'resources/features/test/30/0_2',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/train.csv',
    'out_path': 'resources/matching/train_02_with_test_30_02.csv'
  },
  {
    'matching_name': 'train_05_with_test_30_05',
    'train_path': 'resources/features/train/0_5',
    'test_path': 'resources/features/test/30/0_5',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/train.csv',
    'out_path': 'resources/matching/train_05_with_test_30_05.csv'
  },
  {
    'matching_name': 'train_10_with_test_30_10',
    'train_path': 'resources/features/train/1_0',
    'test_path': 'resources/features/test/30/1_0',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/train.csv',
    'out_path': 'resources/matching/train_10_with_test_30_10.csv'
  },
  {
    'matching_name': 'train_15_with_test_30_15',
    'train_path': 'resources/features/train/1_5',
    'test_path': 'resources/features/test/30/1_5',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/train.csv',
    'out_path': 'resources/matching/train_15_with_test_30_15.csv'
  },
  {
    'matching_name': 'train_02_with_test_50_02',
    'train_path': 'resources/features/train/0_2',
    'test_path': 'resources/features/test/50/0_2',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/train.csv',
    'out_path': 'resources/matching/train_02_with_test_50_02.csv'
  },
  {
    'matching_name': 'train_05_with_test_50_05',
    'train_path': 'resources/features/train/0_5',
    'test_path': 'resources/features/test/50/0_5',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/train.csv',
    'out_path': 'resources/matching/train_05_with_test_50_05.csv'
  },
  {
    'matching_name': 'train_10_with_test_50_10',
    'train_path': 'resources/features/train/1_0',
    'test_path': 'resources/features/test/50/1_0',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/train.csv',
    'out_path': 'resources/matching/train_10_with_test_50_10.csv'
  },
  {
    'matching_name': 'train_15_with_test_50_15',
    'train_path': 'resources/features/train/1_5',
    'test_path': 'resources/features/test/50/1_5',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/train.csv',
    'out_path': 'resources/matching/train_15_with_test_50_15.csv'
  },
]

if __name__ == "__main__":

  # extract mfcc -> dataset to features (npy)
  extract_train(train_data)
  extract_test(test_data)

  # matching dtw
  matching_features(matching_data)

  # evaluate
  evaluation(matching_data)

  pass