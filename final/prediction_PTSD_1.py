import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# data_list = ['relationships', 'assistance',
#              'domesticviolence', 'survivorsofabuse', 'anxiety']
# for data in data_list:
#     # 載入資料
#     df_data = pd.read_csv(f'./datasets/df_{data}_S.csv')

#     # 輸出的資料
#     output_df = pd.DataFrame(columns=['subreddit', 'text', 'label'])

#     # 處理每一行的'text'欄位
#     for text in df_data['text']:
#         # 分割'text'欄位內容
#         # segments = text.split(',')
#         segments = sent_tokenize(text)
#         # 去除每個片段的首尾空白並存儲到新的DataFrame中
#         for segment in segments:
#             # 使用pd.concat來添加新行
#             new_row = pd.DataFrame(
#                 {'subreddit': f'{data}', 'text': [segment.strip()]})
#             output_df = pd.concat([output_df, new_row], ignore_index=True)

#     # 載入模型跟tokenizer
#     model = RobertaForSequenceClassification.from_pretrained(
#         f'./model/model_{data}')
#     tokenizer = RobertaTokenizer.from_pretrained(f'./model/model_{data}')

#     # 資料前處理

#     # 定義獲取預測結果的函數

#     def get_predictions(text, tokenizer, model):
#         # 對新數據進行encoding
#         inputs = tokenizer(text, padding=True, truncation=True,
#                            return_tensors="pt")
#         # 使用模型進行預測
#         with torch.no_grad():
#             outputs = model(**inputs)
#         # 獲取預測的類別
#         predictions = torch.argmax(outputs.logits, dim=1)
#         return predictions.item()

#     predictions = []

#     for text in output_df['text']:
#         prediction = get_predictions(text, tokenizer, model)
#         predictions.append(prediction)
#     output_df['label'] = predictions

#     # 將結果寫入新的CSV檔案
#     output_file = f'./datasets/prediction_{data}_1_.csv'
#     output_df.to_csv(output_file, index=False)

#     print(f"處理完成，結果已儲存在 {output_file}")


# 載入資料
df_data = pd.read_csv(f'./datasets/df_PTSD_N_S.csv')

# 輸出的資料
output_df = pd.DataFrame(columns=['subreddit', 'text', 'label'])

# 處理每一行的'text'欄位
for text in df_data['text']:
    # 分割'text'欄位內容
    # segments = text.split(',')
    segments = sent_tokenize(text)
    # 去除每個片段的首尾空白並存儲到新的DataFrame中
    for segment in segments:
        # 使用pd.concat來添加新行
        new_row = pd.DataFrame(
            {'subreddit': f'ptsd', 'text': [segment.strip()]})
        output_df = pd.concat([output_df, new_row], ignore_index=True)

# 載入模型跟tokenizer
model = RobertaForSequenceClassification.from_pretrained(
    f'./model/model_PTSD')
tokenizer = RobertaTokenizer.from_pretrained(f'./model/model_PTSD')

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

for text in output_df['text']:
    prediction = get_predictions(text, tokenizer, model)
    predictions.append(prediction)
output_df['label'] = predictions

# 將結果寫入新的CSV檔案
output_file = f'./datasets/prediction_PTSD_0_.csv'
output_df.to_csv(output_file, index=False)

print(f"處理完成，結果已儲存在 {output_file}")
