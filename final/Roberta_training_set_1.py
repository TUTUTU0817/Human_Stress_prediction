from sklearn.metrics import precision_score, recall_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from datasets import Dataset, load_dataset

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using CUDA:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device("cpu")
#     print("CUDA is not available. Using CPU instead.")
# 準備資料
# data = 'survivorsofabuse'
df_data = pd.read_csv(f'./datasets/training_set/training_set_2.csv')
data_list = []
df_data['input'] = df_data['subreddit'] + df_data['text']
data_dataset = Dataset.from_pandas(df_data)

# 切分資料集
# train_dataset, val_dataset = train_test_split(encoded_dataset, test_size=0.2)
split_dataset = data_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']


# 加載預訓練的tokenizer和model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', num_labels=2)
# model = RobertaForSequenceClassification.from_pretrained(
#     './model/model_training_RoBERTa_2')
# tokenizer = RobertaTokenizer.from_pretrained(
#     './model/model_training_RoBERTa_2')

# 資料前處理


def preprocess_function(data):
    return tokenizer(data['input'], padding='max_length', truncation=True)


train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 設置模型需要的格式
train_dataset.set_format(type='torch', columns=[
                         'input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=[
                       'input_ids', 'attention_mask', 'label'])
# model.to(device)
# 訓練參數設置
training_args = TrainingArguments(
    output_dir=f'./training_results',
    learning_rate=2e-5,  # 2e-5
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    warmup_steps=100,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    fp16=True
)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    return {'accuracy': acc, 'f1_score': f1, 'precision': precision, 'recall': recall}


# 創建trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
# 訓練模型
trainer.train()


trainer.save_model(f'./model_training_1')
tokenizer.save_pretrained(f'./model_training_1')
print('Model and tokenizer have been saved.')


# 評估模型
eval_results = model.evaluate()
print(eval_results)
print(f"Accuracy: {eval_results['eval_accuracy:']}")
print(f"F1 Score: {eval_results['eval_f1_score:']}")


# from transformers import RobertaTokenizer, RobertaForSequenceClassification

# # 加載最好的模型檢查點
# model_path = f'./{data}_reuslts/checkpoint-177'
# model = RobertaForSequenceClassification.from_pretrained(model_path)
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# # 保存模型和tokenizer
# model.save_pretrained(f'./model/model_{data}')
# tokenizer.save_pretrained(f'./model/model_{data}')

# print('Model and tokenizer have been saved.')
