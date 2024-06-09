import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset, load_dataset

# 準備資料
df_data = pd.read_csv(f'./datasets/merge_data.csv')
models = [
    # 'PTSD',
    # 'relationships',
    'assistance', 'domesticviolence', 'survivorsofabuse', 'anxiety']
data_dataset = Dataset.from_pandas(df_data)

for the_model in models:

    # 加載預訓練的tokenizer和model
    tokenizer = RobertaTokenizer.from_pretrained(f'./model/model_{the_model}')
    model = RobertaForSequenceClassification.from_pretrained(
        f'./model/model_{the_model}')
    # model = RobertaForSequenceClassification.from_pretrained('./model/model_data')
    # tokenizer = RobertaTokenizer.from_pretrained('./model/model_data')

    # 資料前處理

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

    for text in df_data['text']:
        prediction = get_predictions(text, tokenizer, model)
        predictions.append(prediction)
    df_data['label'] = predictions
    # 將結果寫入新的CSV檔案
    output_file = f'./datasets/testing_set_ans_{the_model}.csv'
    df_data.to_csv(output_file, index=False)

    print(f"處理完成，結果已儲存在 {output_file}")
