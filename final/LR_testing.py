import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 加载保存的模型
model_path = './model/logistic_regression_bow_model_2.pkl'
pipeline = joblib.load(model_path)

# 測試數據集合併
# df_ans_anxiety = pd.read_csv(
#     f'./datasets/testing_set_ans/testing_set_ans_anxiety.csv')
# df_ans_assistance = pd.read_csv(
#     f'./datasets/testing_set_ans/testing_set_ans_assistance.csv')
# df_ans_domesticviolence = pd.read_csv(
#     f'./datasets/testing_set_ans/testing_set_ans_domesticviolence.csv')
# df_ans_ptsd = pd.read_csv(
#     f'./datasets/testing_set_ans/testing_set_ans_ptsd.csv')
# df_ans_relationships = pd.read_csv(
#     f'./datasets/testing_set_ans/testing_set_ans_relationships.csv')
# df_ans_survivorsofabuse = pd.read_csv(
#     f'./datasets/testing_set_ans/testing_set_ans_survivorsofabuse.csv')

# df_ans_survivorsofabuse['subreddit'] = 'survivorsofabuse'
# df_ans_anxiety['subreddit'] = 'anxiety'
# df_ans_assistance['subreddit'] = 'assistance'
# df_ans_ptsd['subreddit'] = 'ptsd'
# df_ans_relationships['subreddit'] = 'relationships'
# df_ans_domesticviolence['subreddit'] = 'domesticviolence'


# df_test = pd.concat([df_ans_anxiety, df_ans_assistance, df_ans_domesticviolence,
#                     df_ans_ptsd, df_ans_relationships, df_ans_survivorsofabuse])
# # 删除不需要的列
# df_test.drop(columns=['Unnamed: 2', 'Unnamed: 3'], inplace=True)

# # 将 'subreddit' 列移动到第一列
# subreddit_col = df_test.pop('subreddit')
# df_test.insert(0, 'subreddit', subreddit_col)
# print(df_test)

# df_test.to_csv('./datasets/testing_set_ans/merge_testing_data.csv')

# 读取和准备测试数据
df_test = pd.read_csv(
    f'./datasets/testing_set_ans/merge_testing_data.csv', index_col=0)
# print(df_test)
df_test['input'] = df_test['subreddit'] + df_test['text']

# 文本数据和标签
texts = df_test['input'].values
labels = df_test['label'].values

# 使用加载的模型进行预测
predictions = pipeline.predict(texts)
print(type(predictions), predictions)
# # 评估模型
# accuracy = accuracy_score(labels, predictions)
# f1 = f1_score(labels, predictions, average='weighted')

# print(f"Accuracy: {accuracy}")
# print(f"F1 Score: {f1}")
# print("\nClassification Report:")
# print(classification_report(labels, predictions))

# # 打印预测结果
# df_test['predictions'] = predictions
# print(df_test[['input', 'label', 'predictions']])
