import pandas as pd
import os

df1 = pd.read_csv('C:/Users/Hwang/Desktop/for_merge/opcode.csv')
df2 = pd.read_csv("C:/Users/Hwang/Desktop/for_merge/api.csv")

merge_inner = pd.merge(df1, df2)
#merge_inner

merge_inner.to_csv(r'C:/Users/Hwang/Desktop/for_merge/merge.csv', index = False, header = True)