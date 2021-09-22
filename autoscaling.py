import pandas as pd

print('Please enter path to data (.xls) file')
data = input()
means = pd.read_excel(data).mean(axis=0)
