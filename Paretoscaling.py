# Paretoscaling
import pandas as pd
import numpy as np

# import data
df = pd.read_excel('Data_tocheck.xlsx')                            

# meanscaling
mean_vector = df.mean(axis=0)
mean_centering = np.divide(df, mean_vector)

# calculate std and the square root of std
std_vector = np.std(df, axis=0)
root_std = np.sqrt(std_vector)

# paretoscaling
scaled_data = np.divide(mean_centering, root_std)
