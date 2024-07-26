from sklearn.metrics import precision_score, recall_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from datasets import Dataset, load_dataset

# 檢查cuda是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 資料集名稱列表
data_list = [
            'ptsd', 'relationships', 'domesticviolence', 
             'survivorsofabuse', 
             'anxiety'
             ]
# 載入資料
df_ptsd_0 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_ptsd_0_.csv')
df_ptsd_1 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_ptsd_1_.csv')
df_relationships_0 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_relationships_0_.csv')
df_relationships_1 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_relationships_1_.csv')
df_domesticviolence_0 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_domesticviolence_0_.csv')
df_domesticviolence_1 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_domesticviolence_1_.csv')
df_survivorsofabuse_0 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_survivorsofabuse_0_.csv')
df_survivorsofabuse_1 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_survivorsofabuse_1_.csv')
df_anxiety_0 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_anxiety_0_.csv')
df_anxiety_1 = pd.read_csv(f'./datasets/A_model_prediction/A_model_prediction_anxiety_1_.csv')

# training set 1 -> (1的1) + (0的1) : 1  |  training set 2 -> (1的1) : 1
#                   (1的0) + (0的0) : 0  |                    (0的0) : 0
training_set_1 = pd.concat([df_ptsd_0[df_ptsd_0['label'] == 0], 
                            df_ptsd_1[df_ptsd_1['label'] == 0], 
                            df_ptsd_0[df_ptsd_0['label'] == 1], 
                            df_ptsd_1[df_ptsd_1['label'] == 1],
                            df_relationships_0[df_relationships_0['label'] == 0], 
                            df_relationships_1[df_relationships_1['label'] == 0], 
                            df_relationships_0[df_relationships_0['label'] == 1], 
                            df_relationships_1[df_relationships_1['label'] == 1],
                            df_domesticviolence_0[df_domesticviolence_0['label'] == 0], 
                            df_domesticviolence_1[df_domesticviolence_1['label'] == 0], 
                            df_domesticviolence_0[df_domesticviolence_0['label'] == 1], 
                            df_domesticviolence_1[df_domesticviolence_1['label'] == 1],
                            df_survivorsofabuse_0[df_survivorsofabuse_0['label'] == 0], 
                            df_survivorsofabuse_1[df_survivorsofabuse_1['label'] == 0], 
                            df_survivorsofabuse_0[df_survivorsofabuse_0['label'] == 1], 
                            df_survivorsofabuse_1[df_survivorsofabuse_1['label'] == 1],
                            df_anxiety_0[df_anxiety_0['label'] == 0], 
                            df_anxiety_1[df_anxiety_1['label'] == 0], 
                            df_anxiety_0[df_anxiety_0['label'] == 1], 
                            df_anxiety_1[df_anxiety_1['label'] == 1]])

training_set_2 = pd.concat([df_ptsd_1[df_ptsd_1['label'] == 1],
                            df_ptsd_0[df_ptsd_0['label'] == 0],
                            df_relationships_1[df_relationships_1['label'] == 1],
                            df_relationships_0[df_relationships_0['label'] == 0],
                            df_domesticviolence_1[df_domesticviolence_1['label'] == 1],
                            df_domesticviolence_0[df_domesticviolence_0['label'] == 0],
                            df_survivorsofabuse_1[df_survivorsofabuse_1['label'] == 1],
                            df_survivorsofabuse_0[df_survivorsofabuse_0['label'] == 0],
                            df_anxiety_1[df_anxiety_1['label'] == 1],
                            df_anxiety_0[df_anxiety_0['label'] == 0]])

print("training_set_1的0數量:", training_set_1['label'].value_counts().get(0,0))
print("training_set_1的1數量:", training_set_1['label'].value_counts().get(1,0))
print("training_set_2的0數量:", training_set_2['label'].value_counts().get(0,0))
print("training_set_2的1數量:", training_set_2['label'].value_counts().get(1,0))


# 由於資料不平衡，進行欠採樣
def undersample(df, target_count):
    label_0 = df[df['label'] == 0]
    label_1 = df[df['label'] == 1]
    if len(label_0) > target_count:
        label_0 = label_0.sample(target_count)
    if len(label_1) > target_count:
        label_1 = label_1.sample(target_count)
    return pd.concat([label_0, label_1])

# 確定欠採樣的目標數量
target_count_1 = min(training_set_1['label'].value_counts().get(0,0), training_set_1['label'].value_counts().get(1,0))
target_count_2 = min(training_set_2['label'].value_counts().get(0,0), training_set_2['label'].value_counts().get(1,0))

# 進行欠採樣
training_set_1_balance = undersample(training_set_1, target_count_1)
training_set_2_balance = undersample(training_set_2, target_count_2)

print("training_set_1的0數量:", training_set_1_balance['label'].value_counts().get(0,0))
print("training_set_1的1數量:", training_set_1_balance['label'].value_counts().get(1,0))
print("training_set_2的0數量:", training_set_2_balance['label'].value_counts().get(0,0))
print("training_set_2的1數量:", training_set_2_balance['label'].value_counts().get(1,0))

# 保存到文件
training_set_1_balance.to_csv('./datasets/training_set/training_set_1_balance.csv', index=False)
training_set_2_balance.to_csv('./datasets/training_set/training_set_2_balance.csv', index=False)

print("訓練集保存完成")

