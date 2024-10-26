import json
import pandas as pd
import re


with open("sarcam.json","r",encoding='utf-8') as f:
    data = json.loads(f.read())

print(data[0])

label = [d['is_sarcastic'] for d in data]
headlines = [d['headline'] for d in data]

print(type(label))
print(type(headlines))

for headline in headlines:
    headline = re.sub(r'[\.\,]', '', headline)  # Loại bỏ dấu câu

data_to_scv = {}
data_to_scv['headline']= headlines
data_to_scv['label']= label

df = pd.DataFrame(data_to_scv)

df.to_csv('sarcasm_data.csv', index=False)