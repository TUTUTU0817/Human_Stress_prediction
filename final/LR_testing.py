import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score

# 加載保存的模型
model_path = './model/B_model_LR_bow_model_1.pkl'
pipeline = joblib.load(model_path)


# 資料載入
df_test = pd.read_csv(f'./datasets/testing_set_ans_2/merge_testing_data_2.csv', index_col=0)
df_test['input'] = df_test['subreddit'] + df_test['text']

# 文本和標籤
texts = df_test['input'].values
labels = df_test['label'].values

# 使用加載的模型進行預測
predictions = pipeline.predict(texts)
# print(type(predictions), predictions)

# 評估模型
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average='weighted')
precision = precision_score(labels, predictions, average="weighted")
recall = recall_score(labels, predictions, average="weighted")

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print("\nClassification Report:")
print(classification_report(labels, predictions))

# print預測结果
df_test['predictions'] = predictions
print(df_test[['input', 'label', 'predictions']])




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
