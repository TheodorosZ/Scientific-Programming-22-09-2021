import pandas as pd

data = pd.read_excel('Data_tocheck.xlsx')
mean_vector = data.mean(axis=1)

print(data.shape())

mean_centered_data = 