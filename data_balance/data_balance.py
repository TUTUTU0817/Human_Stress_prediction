import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

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


df_ptsd_count = df_ptsd['label'].value_counts()
df_anxiety_count = df_ptsd['label'].value_counts()
df_domesticviolence_count = df_anxiety['label'].value_counts()
df_relationships_count = df_anxiety['label'].value_counts()
df_survivorsofabuse_count = df_domesticviolence['label'].value_counts()


# 分析各檔案0、1比例
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


# 載入翻譯的模型和分詞器(法文->英文)
back_model_name = f'Helsinki-NLP/opus-mt-{target_lang}-{original_lang}'
back_tokenizer = MarianTokenizer.from_pretrained(back_model_name)
back_model = MarianMTModel.from_pretrained(back_model_name)


def translate(text, tokenizer, model):
    # 編碼 text
    encoded_text = tokenizer.encode(
        text, return_tensors='pt', truncation=True, padding="max_length", max_length=512)
    # 翻譯 text
    translated = model.generate(
        encoded_text, max_length=512, num_beams=5, early_stopping=True)
    # 解碼 text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


# # 示例文本數據
# texts = [
#     "The quick brown fox jumps over the lazy dog",
#     "The customer is very happy",
#     "Data science is fun",
#     "I love machine learning"
# ]


# 使用回譯技術增強文本數據
# for data in datasets:
# augmented_texts = []
# for text in df_ptsd['text'].loc[:10]:
#     translated_text = translate(text, tokenizer, model)
#     back_translated_text = translate(
#         translated_text, back_tokenizer, back_model)
#     augmented_texts.append(back_translated_text)

# print 最後結果
# for i, text in enumerate(augmented_texts):

#     print(f"原始: {df_ptsd['text'].loc[i]}")
#     print(f"增強: {text}\n")
# print(df_ptsd['text'])
