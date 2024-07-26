import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset, load_dataset


# 檢查cuda是否能啟用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 準備資料
df_data = pd.read_csv(f'./datasets/training_set/training_set_2_balance.csv')
df_data['input'] = df_data['subreddit'] + df_data['text']
data_dataset = Dataset.from_pandas(df_data)

# 切分資料集
split_dataset = data_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']


# 加載預訓練的tokenizer和model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
# model = RobertaForSequenceClassification.from_pretrained('./model/model_training_RoBERTa_2')
# tokenizer = RobertaTokenizer.from_pretrained('./model/model_training_RoBERTa_2')
model = model.to(device)

# 資料前處理
def preprocess_function(data):
    return tokenizer(data['input'], padding='max_length', truncation=True)


train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 設置模型需要的格式
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# 訓練參數設置
training_args = TrainingArguments(
    output_dir=f'./B_Roberta_model_training_results',
    learning_rate=2e-5,  # 2e-5
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

# 計算評估指標函數
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

# 儲存模型
trainer.save_model(f'./model/B_model_Roberta_2')
tokenizer.save_pretrained(f'./B_model_Roberta_2_tokenizer')
print('Model and tokenizer have been saved.')


# 評估模型
eval_results = trainer.evaluate()
print(f"Evaluation results for B_model_Roberta_2:")
print(eval_results)
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1 Score: {eval_results['eval_f1_score']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")


# from transformers import RobertaTokenizer, RobertaForSequenceClassification

# # 加載最好的模型檢查點
# model_path = f'./{data}_reuslts/checkpoint-177'
# model = RobertaForSequenceClassification.from_pretrained(model_path)
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# # 保存模型和tokenizer
# model.save_pretrained(f'./model/model_{data}')
# tokenizer.save_pretrained(f'./model/model_{data}')

# print('Model and tokenizer have been saved.')
