from optionCombo import optionModel,preInit
import pandas as pd
import numpy as np
from datetime import datetime
option_data = pd.read_csv('../test/optionsD.csv',dtype={
    'is_call':str,
    'K':float,
    'askIV':float,
    'bidIV':float,
    'expiry':float,
    'expiration':str
})
option_data['expiration'] = pd.to_datetime(option_data['expiration'])
pricedata = pd.read_csv('../test/price.csv')
expirDate = '2023-06-30T00:00:00.000000000'
pricedata['Timestamp'] = pd.to_datetime(pricedata['Timestamp'])
pricedata["close"] = pd.to_numeric(pricedata["close"])
print(pricedata['Timestamp'].diff())
preOption,joined1,price = preInit.Prep( expirDate, optionDf = option_data, priceDf = pricedata, interval=3, strikePriceRange=0.2)
model = optionModel.option_model(price, joined1,preOption,optiontypes = [[1,1,1]],tradetypes=[[1,-1,1]], maxquantity=1)  # Init
df = model.options_model_finder()
for para in df.para[:1]:
    model.model_plot(para)
