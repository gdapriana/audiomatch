import numpy as np
import time

from core_functions.dtw.dtw import fastdtw
def compare(trains, tests):
  result = []
  for test in tests:
    all_distance = []
    start_exec = time.time()
    for train in trains:
      x = np.load(train['npy_path'])
      y = np.load(test['npy_path'])
      distance, _ = fastdtw(x, y)
      all_distance.append({ 'train_title': train['title'], 'train_artist': train['artist'], 'distance': round(float(distance), 2) })
    end_exec = round((time.time() - start_exec), 2)
    min_distance = min(all_distance, key=lambda data: data['distance'])
    min_distance['exec_time'] = end_exec
    result.append({
      'title': test['title'],
      'predicted_title': min_distance['train_title'],
      'predicted_artist': min_distance['train_artist'],
      'dtw_score': min_distance['distance'],
      'exec_time': min_distance['exec_time'],
    })
  return result
