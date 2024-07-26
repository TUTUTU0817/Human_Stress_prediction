import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset
import torch
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# 檢查是否可用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_list = [
            # 'ptsd',
            # 'relationships', 
            # 'domesticviolence', 
            'survivorsofabuse',
            #  'anxiety'
             ]
for data in data_list:
    # 載入資料
    df_data = pd.read_csv(f'./data_balance/balance_{data}.csv')
    df_data = df_data.drop(columns=['post_id', 'sentence_range', 'confidence', 'social_timestamp'])
    dataset = Dataset.from_pandas(df_data)

    # 載入模型跟tokenizer (訓練A model)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = model.to(device)

    # 切割數據集
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    # 資料前處理函數
    def preprocess_function(data):
        return tokenizer(data['text'], padding='max_length', truncation=True)

    # 資料格式設置
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    train_dataset.set_format(type='torch', columns=[
                             'input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=[
                           'input_ids', 'attention_mask', 'label'])

    # 訓練參數設置
    training_args = TrainingArguments(
        output_dir=f'./results/{data}_results',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )

    # 計算評估指標函數 
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average='weighted')
        acc = accuracy_score(p.label_ids, preds)
        pre = precision_score(p.label_ids, preds, average='weighted')
        recall = recall_score(p.label_ids, preds, average='weighted')
        return {'accuracy': acc, 'f1_score': f1, 'precision': pre, 'recall': recall}

    # 創建Trainer
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

    # 保存model和tokenizer
    trainer.save_model(f'./model/A_model_{data}')
    tokenizer.save_pretrained(f'./model/A_model_{data}')
    print(f'Model and tokenizer for {data} have been saved.')

    # 評估模型
    eval_results = trainer.evaluate()
    print(f"Evaluation results for {data}:")
    print(eval_results)
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1_score']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall: {eval_results['eval_recall']:.4f}")
