import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, load_dataset

# 定义要运行的数据集列表
datasets_list = ['./datasets/df_relationships.csv', './datasets/df_assistance.csv',
                 './datasets/df_domesticviolence.csv', './datasets/df_survivorsofabuse.csv', './datasets/df_anxiety.csv']

# 遍历数据集列表
for dataset_file in datasets_list:
    # 加载数据集
    df = pd.read_csv(dataset_file)
    dataset_file = dataset_file[14:-4]
    print(dataset_file)
    dataset = Dataset.from_pandas(df)

    # 切分数据集
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    # 加载预训练的 tokenizer 和 model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2)

    # 数据预处理函数
    def preprocess_function(data):
        return tokenizer(data['text'], padding='max_length', truncation=True)

    # 数据格式设置
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    train_dataset.set_format(type='torch', columns=[
                             'input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=[
                           'input_ids', 'attention_mask', 'label'])

    # 训练参数设置
    training_args = TrainingArguments(
        output_dir=f'./{dataset_file}_results',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )

    # 计算评估指标的函数
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average='weighted')
        acc = accuracy_score(p.label_ids, preds)
        return {'accuracy': acc, 'f1_score': f1}

    # 创建 Trainer 实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 训练模型
    trainer.train()

    # 保存模型和 tokenizer
    trainer.save_model(f'./model/model_{dataset_file}')
    tokenizer.save_pretrained(f'./model/model_{dataset_file}')
    print(f'Model and tokenizer for {dataset_file} have been saved.')

    # 评估模型
    eval_results = trainer.evaluate()
    print(f"Evaluation results for {dataset_file}:")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1_score']:.4f}")
