import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch

# 檢查是否可用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 資料
# ptsd
df_ptsd = pd.read_csv("./datasets/original/df_PTSD.csv")
# anxiety
df_anxiety = pd.read_csv("./datasets/original/df_anxiety.csv")
# domesticviolence
df_domesticviolence = pd.read_csv(
    "./datasets/original/df_domesticviolence.csv")
# relationship
df_relationships = pd.read_csv("./datasets/original/df_relationships.csv")
# sirvivoesofabuse
df_survivorsofabuse = pd.read_csv(
    "./datasets/original/df_survivorsofabuse.csv")

# 計算資料筆數
df_ptsd_count = df_ptsd['label'].value_counts()
df_anxiety_count = df_ptsd['label'].value_counts()
df_domesticviolence_count = df_anxiety['label'].value_counts()
df_relationships_count = df_anxiety['label'].value_counts()
df_survivorsofabuse_count = df_domesticviolence['label'].value_counts()

# label=1,label=0之差
ptsd_diff = df_ptsd_count.get(1, 0) - df_ptsd_count.get(0, 0)
anxiety_diff = df_anxiety_count.get(1, 0) - df_anxiety_count.get(0, 0)
domesticviolence_diff = df_domesticviolence_count.get(1, 0) - df_domesticviolence_count.get(0, 0)
relationships_diff = df_relationships_count.get(1, 0) - df_relationships_count.get(0, 0)
survivorsofabuse_diff = df_survivorsofabuse_count.get(1, 0) - df_survivorsofabuse_count.get(0, 0)

# 印出1,0分別數量
print("ptsd的1 : ", df_ptsd_count.get(1, 0))
print("ptsd的0 : ", df_ptsd_count.get(0, 0))
print("anxiety的1 : ", df_anxiety_count.get(1, 0))
print("anxiety的0 : ", df_anxiety_count.get(0, 0))
print("domesticviolence的1 : ",
      df_domesticviolence_count.get(1, 0))
print("domesticviolence的0 : ",
      df_domesticviolence_count.get(0, 0))
print("relationships的1 : ", df_relationships_count.get(1, 0))
print("relationships的0 : ", df_relationships_count.get(0, 0))
print("survivorsofabuse的1 : ",
      df_survivorsofabuse_count.get(1, 0))
print("survivorsofabuse的0 : ",
      df_survivorsofabuse_count.get(0, 0))


# 載入翻譯的模型和分詞器(英文->法文)
original_lang = 'en'
target_lang = 'fr'
model_name = f'Helsinki-NLP/opus-mt-{original_lang}-{target_lang}'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
model = model.to(device)

# 載入翻譯的模型和分詞器(法文->英文)
back_model_name = f'Helsinki-NLP/opus-mt-{target_lang}-{original_lang}'
back_tokenizer = MarianTokenizer.from_pretrained(back_model_name)
back_model = MarianMTModel.from_pretrained(back_model_name)
back_model = back_model.to(device)

# 將文本進行翻譯的函數
def translate(text, tokenizer, model):
    # 編碼 text
    encoded_text = tokenizer.encode(
        text, return_tensors='pt', truncation=True, padding="max_length", max_length=512).to(device)
    # 翻譯 text
    translated = model.generate(
        encoded_text, max_length=512, num_beams=5, early_stopping=True).to(device)
    # 解碼 text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


# 示例文本數據
# texts = [
#     "The quick brown fox jumps over the lazy dog",
#     "The customer is very happy",
#     "Data science is fun",
#     "I love machine learning"
# ]

# 由於要將0的數量補足到與1一樣，取出0的資料進行翻譯擴增
df_ptsd_0 = df_ptsd[df_ptsd['label'] == 0]
df_anxiety_0 = df_anxiety[df_anxiety['label'] == 0]
df_relationships_0 = df_relationships[df_relationships['label'] == 0]
df_domesticviolence_0 = df_domesticviolence[df_domesticviolence['label'] == 0]
df_survivorsofabuse_0 = df_survivorsofabuse[df_survivorsofabuse['label'] == 0]




# 定義回譯增強文本數據函式
def augment_text(df, tokenizer, model, back_tokenizer, back_model, diff):
    # 存放增強後的數據
    augmented_texts = []
    for text in df['text'].loc[:diff]:
        translated_text = translate(text, tokenizer, model)
        back_translated_text = translate(
            translated_text, back_tokenizer, back_model)
        augmented_texts.append(back_translated_text)
    print(diff, len(augmented_texts))
    return augmented_texts


# 利用迴圈將每個資料進行擴增
datasets = [
            "df_ptsd_0", 
            "df_anxiety_0","df_relationships_0","df_domesticviolence_0","df_survivorsofabuse_0"
            ]
# 對應的 diff_index 字典
diff_indices = {
    "df_ptsd_0": ptsd_diff,
    "df_anxiety_0": anxiety_diff,
    "df_relationships_0": relationships_diff,
    "df_domesticviolence_0": domesticviolence_diff,
    "df_survivorsofabuse_0": survivorsofabuse_diff
}

# 翻譯擴增
for df_name in datasets:
    # 當df_name是字串，會在全域中找到該字串的變數並返回變數值
    df_original = globals()[f"df_{df_name[3:-2]}"]
    df_0 = globals()[df_name]
    diff = diff_indices[df_name]    
    augmented_texts = augment_text(df_0, tokenizer, model, back_tokenizer, back_model, 300)

    # 將結果寫入新的CSV檔案
    output_file = f'./data_balance/balance_{df_name[3:-2]}.csv'
    augment_df = pd.DataFrame({"subreddit":df_name[3:-2], "text":augmented_texts, 'label':0})
    balance_df = pd.concat([df_original, augment_df[:diff]])
    balance_df.to_csv(output_file, index=False)
    print(f"處理完成，結果已儲存在 {output_file}")
    


# print 最後結果
# for i, text in enumerate(augmented_texts):

#     print(f"原始: {df_ptsd['text'].loc[i]}")
#     print(f"增強: {text}\n")


# df = pd.read_csv('./data_balance/balance_ptsd.csv')
# df_count = df['label'].value_counts()
# diff = df_count.get(1, 0) - df_count.get(0, 0)
# print("原本差距" , ptsd_diff)
# print("後來的1", df_count.get(1, 0))
# print("後來的0", df_count.get(0, 0))
# print("後來的差距", diff)

