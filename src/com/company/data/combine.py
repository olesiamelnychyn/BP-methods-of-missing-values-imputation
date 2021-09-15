#src: https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
import os
import glob
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

os.chdir("/Users/olesia/IdeaProjects/BP-firts-attempt/src/com/company/data/shmu_Open_data_1_upr")

all_filenames = [i for i in glob.glob('*.{}'.format('csv'))]

#combine all files in the list
combined_csv = DataFrame()
n=len(all_filenames)
for f in all_filenames[:n]:
    print(str(all_filenames.index(f))+" / "+str(n-1))
    df =pd.read_csv(f, header = None, delimiter=";")
    df = df.loc[df[1] == 'ASBA'].drop(columns=[0,1, 2,3,4,5,13])
    combined_csv = pd.concat([combined_csv, df])

#export to csv
combined_csv.sort_values(by=[6]).drop(columns=[6]).to_csv( "../combined_csv_new.csv", index=False, header=None, encoding='utf-8-sig')