import pandas as pd
def evaluation(test_result):
  result = []

  for test in test_result:
    print(f"evaluating {test['matching_name']} result")
    tp = 0
    fn = 0
    exec_time = []

    csv_result = pd.read_csv(test['out_path'])
    for _, row in csv_result.iterrows():
      if row['title'] == row['predicted_title']: tp += 1
      else: fn += 1
      exec_time.append(row['exec_time'])

    accuracy = tp / (tp + fn) * 100
    exec_average = sum(exec_time) / len(exec_time)
    test_name = test['matching_name']
    result.append({
      'test_name': test_name,
      'accuracy (%)': round(accuracy, 2),
      'execution_average (s)': round(exec_average, 2),
    })
  pd.DataFrame(result).to_csv("resources/evaluation/evaluation.csv", index=False)
  print(f"Done, result in in resources/evaluation/evaluation.csv")





