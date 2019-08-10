#Our imports
import math
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.code import Dense, Activation, Dropout

#Fix random seed for reproducibility
np.random.seed(42)

data = pd.read_csv("../Data/DIS.csv")