from core_functions.evaluation.evaluation import evaluation
from suport_function.extract import extract_train, extract_test
from suport_function.matching import matching_features

train_data = [
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
    'duration_50': [
      {
        'csv_path': 'resources/csv/test_normal_50.csv',
        'params': {'frame_size': 0.5, 'frame_hop': 0.25, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/50',
        'out_path': 'resources/features/test/50/0_5'
      },
      {
        'csv_path': 'resources/csv/test_normal_50.csv',
        'params': {'frame_size': 1.0, 'frame_hop': 0.5, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/50',
        'out_path': 'resources/features/test/50/1_0'
      },
      {
        'csv_path': 'resources/csv/test_normal_50.csv',
        'params': {'frame_size': 1.5, 'frame_hop': 0.75, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/50',
        'out_path': 'resources/features/test/50/1_5'
      },
    ],
    'duration_100': [
      {
        'csv_path': 'resources/csv/test_normal_100.csv',
        'params': {'frame_size': 0.5, 'frame_hop': 0.25, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/100',
        'out_path': 'resources/features/test/100/0_5'
      },
      {
        'csv_path': 'resources/csv/test_normal_100.csv',
        'params': {'frame_size': 1.0, 'frame_hop': 0.5, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/100',
        'out_path': 'resources/features/test/100/1_0'
      },
      {
        'csv_path': 'resources/csv/test_normal_100.csv',
        'params': {'frame_size': 1.5, 'frame_hop': 0.75, 'emphasis': 0.97, 'n_fft': 512, 'n_mels': 40, 'n_mfcc': 13},
        'audio_path': 'resources/dataset/test/100',
        'out_path': 'resources/features/test/100/1_5'
      },
    ]
}

matching_data = [
  {
    'matching_name': 'train_05_with_test_100_05',
    'train_path': 'resources/features/train/0_5',
    'test_path': 'resources/features/test/100/0_5',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/test_normal_100.csv',
    'out_path': 'resources/matching/train_05_with_test_100_05.csv'
  },
  {
    'matching_name': 'train_05_with_test_50_05',
    'train_path': 'resources/features/train/0_5',
    'test_path': 'resources/features/test/50/0_5',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/test_normal_50.csv',
    'out_path': 'resources/matching/train_05_with_test_50_05.csv'
  },
  {
    'matching_name': 'train_10_with_test_100_10',
    'train_path': 'resources/features/train/1_0',
    'test_path': 'resources/features/test/100/1_0',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/test_normal_100.csv',
    'out_path': 'resources/matching/train_10_with_test_100_10.csv'
  },
  {
    'matching_name': 'train_10_with_test_50_10',
    'train_path': 'resources/features/train/1_0',
    'test_path': 'resources/features/test/50/1_0',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/test_normal_50.csv',
    'out_path': 'resources/matching/train_10_with_test_50_10.csv'
  },
  {
    'matching_name': 'train_15_with_test_100_15',
    'train_path': 'resources/features/train/1_5',
    'test_path': 'resources/features/test/100/1_5',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/test_normal_100.csv',
    'out_path': 'resources/matching/train_15_with_test_100_15.csv'
  },
  {
    'matching_name': 'train_15_with_test_50_15',
    'train_path': 'resources/features/train/1_5',
    'test_path': 'resources/features/test/50/1_5',
    'train_csv': 'resources/csv/train.csv',
    'test_csv': 'resources/csv/test_normal_50.csv',
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