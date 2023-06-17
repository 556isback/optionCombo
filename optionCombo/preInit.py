import pandas as pd
from optionCombo.func import *
import pandas as pd
import py_vollib.black_scholes as bs
import py_vollib.black_scholes.greeks.numerical as greeks
import py_vollib_vectorized
from datetime import datetime , timezone, timedelta
import numpy as np

def Prep(time1, optionDf, priceDf = None, spotPrice = None,interval=2,Bound=None,strikePriceRange=0.3,impiledVolRange=0.3):

    daysTillExpir = (pd.to_datetime(time1) - datetime.now())/pd.to_timedelta('1d')

    if not spotPrice:
        if type(priceDf) != pd.core.frame.DataFrame:
            raise Exception('please provide either spotPrice or DataFrame object of price data')
        else:
            spotPrice = priceDf.close.values[-1]

    if not Bound:
        if type(priceDf) != pd.core.frame.DataFrame:
            raise Exception('please provide either price Bound or DataFrame object of price data')

    lowerB, upperB = None,None

    if not Bound:
        lowerB,upperB = calRange(priceDf,daysTillExpir,interval)
    elif len(Bound) == 2:
        lowerB,upperB = min(Bound),max(Bound)

    if lowerB and upperB:
        lowerK,upperK = 1-strikePriceRange/2,1+strikePriceRange/2

        lowerIV,upperIV = 1-impiledVolRange/2,1+impiledVolRange/2



        option_data = optionDf.loc[
            (optionDf['expiration'] == time1) & (optionDf['K'] <= spotPrice * upperK) & (optionDf['K'] >= spotPrice * lowerK)
        ]

        option_data['per_ivs'] = option_data['bidIV'].apply(lambda x: list(np.linspace(1 * lowerIV, 1 * upperIV, 89)), 0)


        temp = option_data.explode(['per_ivs'])

        joined1 = temp
        spotPrices = np.linspace(lowerB, upperB, 89)
        spotPer = np.linspace(1 * lowerB / spotPrice, 1 * upperB / spotPrice, 89)
        spotPrices = pd.DataFrame({'spot_price': spotPrices, 'spot_per': spotPer})

        joined1 = joined1.join(spotPrices, how='cross')

        risk_free_rate = 0.02

        joined1['buy_price'] = \
        bs.black_scholes(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                         joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_price'] = \
        bs.black_scholes(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                         joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_delta'] = \
        greeks.delta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                     joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_delta'] = \
        greeks.delta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                     joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_vega'] = \
        greeks.vega(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                    joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_vega'] = \
        greeks.vega(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                    joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_gamma'] = \
        greeks.gamma(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                     joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_gamma'] = \
        greeks.gamma(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                     joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_theta'] = \
        greeks.theta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                     joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_theta'] = \
        greeks.theta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expiry'], risk_free_rate,
                     joined1['bidIV']).values.reshape(1, -1)[0]

        preOption1 = {'C': {}, 'P': {}}
        d = list(joined1.groupby(['is_call']))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                preOption1[type1[0]][stri[0]] = stri[1][
                    ['buy_price', 'sell_price', 'spot_price', 'buy_delta', 'sell_delta', 'buy_vega', 'sell_vega',
                     'buy_gamma', 'sell_gamma', 'buy_theta', 'sell_theta', 'spot_per', 'per_ivs']]


        joined1 = temp
        joined1 = joined1.join(spotPrices, how='cross')

        expirys = [0]
        # expirys = pd.DataFrame({'expirys':expirys})
        # joined1 = joined1.join(expirys,how='cross')
        risk_free_rate = 0.02

        joined1['buy_price'] = \
        bs.black_scholes(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                         joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_price'] = \
        bs.black_scholes(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                         joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_delta'] = \
        greeks.delta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                     joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_delta'] = \
        greeks.delta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                     joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_vega'] = \
        greeks.vega(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                    joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_vega'] = \
        greeks.vega(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                    joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_gamma'] = \
        greeks.gamma(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                     joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_gamma'] = \
        greeks.gamma(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                     joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_theta'] = \
        greeks.theta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                     joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_theta'] = \
        greeks.theta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], expirys, risk_free_rate,
                     joined1['bidIV']).values.reshape(1, -1)[0]

        preOption2 = {'C': {}, 'P': {}}
        d = list(joined1.groupby(['is_call']))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                preOption2[type1[0]][stri[0]] = stri[1][
                    ['buy_price', 'sell_price', 'spot_price', 'buy_delta', 'sell_delta', 'buy_vega', 'sell_vega',
                     'buy_gamma', 'sell_gamma', 'buy_theta', 'sell_theta', 'spot_per', 'per_ivs']]

        preOption3 = {'C': {}, 'P': {}}
        d = list(option_data.groupby(['is_call']))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                preOption3[type1[0]][stri[0]] = stri[1][
                    ['askIV', 'bidIV','expiry']]

        return [preOption1,preOption2,preOption3],joined1,spotPrice
    else:
        raise ValueError(" incorrect form of price bound")