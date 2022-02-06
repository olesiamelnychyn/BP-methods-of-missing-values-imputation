#src: https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
import pandas as pd
import random

    
f="/Users/olesia/IdeaProjects/BP-firts-attempt/src/com/company/data/combined_complete.csv"
df =pd.read_csv(f, header = None, delimiter=",")
for idx, row in df.iterrows():
    if random.randint(1,4) == 1:
        df.iat[idx, random.randint(0,5)]= None

# #export to csv
df.to_csv( "../src/com/company/data/combined_missing.csv", index=False, header=None, encoding='utf-8')