import pandas as pd
import numpy as np

print('=== 最终集成学习 - 融合所有模型 ===')

# 读取所有可用的提交文件
submission_files = [
    'keras_cnn_submission.csv',
    'cnn_submission.csv', 
    'submission.csv',
    'ensemble_vote_submission.csv'
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
submission.to_csv('final_ensemble_submission.csv', index=False)

print('完成！提交文件: final_ensemble_submission.csv')

# 统计每个提交文件的差异
print('\n各提交文件对比:')
for i, file in enumerate(submission_files):
    df = pd.read_csv(file)
    diff = sum(df['Label'].values != final_preds)
    print(f'  {file}: 与最终结果差异 {diff} 个样本')