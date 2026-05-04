import pandas as pd
import numpy as np

print('=== 集成学习 - 基于现有提交文件 ===')

# 读取所有可用的提交文件
submission_files = [
    'cnn_submission.csv',
    'submission.csv',
    'keras_cnn_submission.csv',
    'ensemble_vote_submission.csv',
    'final_ensemble_submission.csv'
]

print(f'加载 {len(submission_files)} 个提交文件...')
predictions = []
for file in submission_files:
    df = pd.read_csv(file)
    preds = df['Label'].values
    predictions.append(preds)
    print(f'  {file}: {len(preds)} 预测')

# 多数投票
print('\n进行投票...')
final_preds = []
for i in range(len(predictions[0])):
    votes = [pred[i] for pred in predictions]
    counts = np.bincount(votes)
    final_pred = np.argmax(counts)
    final_preds.append(final_pred)

# 创建提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(final_preds) + 1),
    'Label': final_preds
})
submission.to_csv('keras_cnn_submission.csv', index=False)

print('完成！提交文件已更新: keras_cnn_submission.csv')

# 统计差异
print('\n各提交文件与最终结果的差异:')
for i, file in enumerate(submission_files):
    df = pd.read_csv(file)
    diff = sum(df['Label'].values != final_preds)
    print(f'  {file}: {diff} 个差异')