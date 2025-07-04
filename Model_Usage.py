from joblib import load, dump
import numpy as np

#WE CAN JUST USE THE MODEL WE MADE BY US BEFORE EASILY BY LOADING IT
model= load("Dragon.joblib")

features=np.array([[0.02985, 0.00, 2.180, 0, 0.4580, 6.4300, 58.70, 6.0622, 3, 222.0, 18.70, 394.12, 5.21]])
predictions= model.predict(features)
print(f"Median value of owner-occupied homes in $1000's: {predictions}")
