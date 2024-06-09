import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# # 準備資料
# df_data = pd.read_csv(f'./datasets/training_set/training_set_2.csv')
# df_data['input'] = df_data['subreddit'] + df_data['text']

# # 文本数据和标签
# texts = df_data['input'].values
# labels = df_data['label'].values

# # # 标签编码
# # label_encoder = LabelEncoder()
# # labels = label_encoder.fit_transform(labels)

# # 切分資料集
# X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# # 创建TF-IDF向量器和逻辑回归模型的pipeline
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
#     ('lr', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000))
# ])

# # 训练模型
# pipeline.fit(X_train, y_train)

# # 评估模型
# y_pred = pipeline.predict(X_val)
# accuracy = accuracy_score(y_val, y_pred)
# f1 = f1_score(y_val, y_pred, average='weighted')

# print(f"Accuracy: {accuracy}")
# print(f"F1 Score: {f1}")
# print("\nClassification Report:")
# print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# # 保存模型
# model_path = './model/logistic_regression_model.pkl'
# joblib.dump(pipeline, model_path)
# print(f'Model has been saved to {model_path}')


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

# 準備資料
df_data = pd.read_csv(f'./datasets/training_set/training_set_2.csv')
df_data['input'] = df_data['subreddit'] + df_data['text']

# 文本数据和标签
texts = df_data['input'].values
labels = df_data['label'].values  # 0和1标签

# 切分資料集
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# 创建BOW向量器和逻辑回归模型的pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('lr', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000))
])

# 训练模型
pipeline.fit(X_train, y_train)

# 评估模型
y_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred,
      target_names=['Class 0', 'Class 1']))

# 保存模型
model_path = './model/logistic_regression_bow_model_2.pkl'
joblib.dump(pipeline, model_path)
print(f'Model has been saved to {model_path}')
