# Paretoscaling
import pandas as pd
import numpy as np
import math

#df =                             #some sort of dataframe

mean_vector = df.mean(axis=1)
mean_centering = np.devide(df, mean_vector)

std_vector = np.std(df)
root_std = math.sqrt(std_vector)

scaled_data = np.devide(mean_centering, root_std)