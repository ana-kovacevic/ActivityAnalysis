import pandas as pd

data = pd.read_csv('activities_out.csv')

#data = pd.read_csv('C:\\Users\\Ana Kovacevic\\Documents\\DataForActivityAnalysis\\activities_out.csv')

data['Normalised'] = data.groupby(['user_in_role_id', 'detection_variable_name'])['measure_value'].apply(lambda x: (x-x.min())/(x.max()-x.min()))

data.to_csv("D:/Posao/Projekti/City4Age/activities_out.csv")


#!!! Align data to 0 time

'''
# Dynamic time warping

!pip install fastdtw
import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
y = np.array([[2,2], [3,3], [4,4]])
distance, path = fastdtw(x, y, dist=)
print(distance)

fastdtw(x,y,dist=)
'''