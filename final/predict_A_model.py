import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# 檢查是否能啟用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 資料集名稱列表
data_list = [
            'ptsd', 'relationships', 'domesticviolence', 
             'survivorsofabuse', 
             'anxiety'
             ]

for data in data_list:

    # 載入資料
    df_data = pd.read_csv(f'./data_balance/balance_{data}.csv')
    df_data = df_data[df_data['label'] == 0]
    # 輸出的資料
    output_df = pd.DataFrame(columns=['subreddit', 'text', 'label'])

    # 將常句子切割成短句子
    for text in df_data['text']:
        segments = sent_tokenize(text)
         # 去除每個片段的首尾空白並存儲到新的DataFrame中
        for segment in segments:
            # 使用pd.concat來添加新行
            new_row = pd.DataFrame(
                {'subreddit': f'{data}', 'text': [segment.strip()]})
            output_df = pd.concat([output_df, new_row], ignore_index=True)
    # 載入模型跟tokenizer
    model = RobertaForSequenceClassification.from_pretrained(f'./model/A_model_{data}')
    # model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained(f'./model/A_model_{data}')
    model = model.to(device)

    # 定義獲取預測結果的函數
    def get_predictions(text, tokenizer, model):
        # 對新數據進行encoding
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        # 使用模型進行預測
        with torch.no_grad():
            outputs = model(**inputs)
        # 獲取預測的類別
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.item()
    
    # 預測結果存放
    predictions = []


    for text in output_df['text']:
        prediction = get_predictions(text, tokenizer, model)
        predictions.append(prediction)
    output_df['label'] = predictions



    # 將結果寫入新的CSV檔案
    output_file = f'./datasets/A_model_prediction/A_model_prediction_{data}_0_.csv'
    output_df.to_csv(output_file, index=False)

    print(f"處理完成，結果已儲存在 {output_file}")
