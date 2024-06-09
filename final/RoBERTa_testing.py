import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 載入資料
# df_test = pd.read_csv(
#     f'./datasets/testing_set_ans/merge_testing_data.csv', index_col=0)
df_test = pd.read_csv(
    f'./datasets/testing_set_ans/merge_testing_data.csv', index_col=0)
# print(df_test)
df_test['input'] = df_test['subreddit'] + df_test['text']

# 載入模型跟tokenizer
model = RobertaForSequenceClassification.from_pretrained(
    f'./model/model_training_RoBERTa_1')
tokenizer = RobertaTokenizer.from_pretrained(
    f'./model/model_training_RoBERTa_1')
# model = RobertaForSequenceClassification.from_pretrained(
#     f'./model/model_training_RoBERTa')
# tokenizer = RobertaTokenizer.from_pretrained(
#     f'./model/model_training_RoBERTa')


# 文本数据和标签
texts = df_test['input'].values
labels = df_test['label'].values


# 定義獲取預測結果的函數


def get_predictions(text, tokenizer, model):
    # 對新數據進行encoding
    inputs = tokenizer(text, padding=True, truncation=True,
                       return_tensors="pt")
    # 使用模型進行預測
    with torch.no_grad():
        outputs = model(**inputs)
    # 獲取預測的類別
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()


predictions = []
for text in texts:
    prediction = get_predictions(text, tokenizer, model)
    predictions.append(prediction)
# 評估模型
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("\nClassification Report:")
print(classification_report(labels, predictions))
