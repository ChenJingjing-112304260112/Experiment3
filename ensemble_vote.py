import pandas as pd
import numpy as np

print('=== 集成学习 - 投票法 ===')

# 读取所有提交文件
submissions = []
submissions.append(pd.read_csv('keras_cnn_submission.csv'))
submissions.append(pd.read_csv('cnn_submission.csv'))
submissions.append(pd.read_csv('submission.csv'))

print(f'加载了 {len(submissions)} 个提交文件')

# 获取所有预测结果
predictions = []
for i, sub in enumerate(submissions):
    preds = sub['Label'].values
    predictions.append(preds)
    print(f'提交文件 {i+1}: {len(preds)} 个预测')

# 投票法：选择每个样本出现次数最多的预测
print('进行投票...')
final_preds = []
for i in range(len(predictions[0])):
    votes = [pred[i] for pred in predictions]
    # 使用众数作为最终预测
    final_pred = max(set(votes), key=votes.count)
    final_preds.append(final_pred)

# 创建集成提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(final_preds) + 1),
    'Label': final_preds
})

submission.to_csv('ensemble_vote_submission.csv', index=False)
print('集成提交文件已保存: ensemble_vote_submission.csv')
print('=== 完成 ===')