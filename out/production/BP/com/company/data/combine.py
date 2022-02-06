#src: https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
import os
import glob
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from datetime import datetime


os.chdir("/Users/olesia/IdeaProjects/BP-firts-attempt/src/com/company/data/shmu_Open_data_1_upr")

all_filenames = [i for i in glob.glob('*.{}'.format('csv'))]
date_zero = datetime.strptime('2016-01-01_00', '%Y-%m-%d_%H')
#combine all files in the list
combined_csv = DataFrame()
all_filenames.sort()
n=len(all_filenames)
for f in all_filenames[:n]:
    print(str(all_filenames.index(f))+" / "+str(n-1))
    df =pd.read_csv(f, header = None, delimiter=";")
    df = df.loc[df[1] == 'ASBA'].drop(columns=[0,1,2,3,4,5,10,13])
    df = df[df[7]!="  null"]
    df[6]=int((datetime.strptime(f.split(".")[0], '%Y-%m-%d_%H') - date_zero).total_seconds()/3600)
    combined_csv = pd.concat([combined_csv, df])

#export to csv
combined_csv.to_csv( "../combined_complete.csv", index=False, header=None, encoding='utf-8')