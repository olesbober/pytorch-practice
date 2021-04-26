# This model will attempt to guess the closing price of bitcoin based on:
# - opening price
# - highest price of the day
# - lowest price of the day
# - volume
# - market cap
# I will try to use PyTorch and Tensors wherever possible.

import numpy as np
import pandas as pd
import torch as tr

class BitCoinModel():

    # read the data
    def __init__(self):
        data = pd.read_csv('BitCoin.csv')
        self.dates = data.iloc[:,1].to_numpy()
        self.open = data.iloc[:,2].to_numpy()
        self.high = data.iloc[:,3].to_numpy()
        self.low = data.iloc[:,4].to_numpy()
        self.close = data.iloc[:,5].to_numpy()
        self.volume = data.iloc[:,6].to_numpy()
        self.market_cap = data.iloc[:,7].to_numpy()

# main
model = BitCoinModel()