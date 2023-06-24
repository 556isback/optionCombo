import pandas as pd
from optionCombo.func import *
import pandas as pd
import py_vollib.black_scholes as bs
import py_vollib.black_scholes.greeks.numerical as greeks
import py_vollib_vectorized
from datetime import datetime , timezone, timedelta
import numpy as np

def Prep(expiryDate, optionDf, priceDf = None, spotPrice = None, interval = 2, Bound = None ,strikePriceRange = 0.3 , impiledVolRange = 0.3 ):

    """
    this function is for pre-computation, like option delta, vega, theta, and option price.
    it calculate the above numbers in a preset range eg. impiledVolRange, price range.
    it also calculate the above numbers when option expiry.

    :param expiryDate: expiry date of the option you want to trade on, doesn't support multiple dates at the moment
    :param optionDf: DataFrame containing option's expiry date, bid ask Iv, strike price, and option type
    :param priceDf: DataFrame containing underlying asset's price data, it is used for determine expected price range and get current asset price, it's not necessary when privided spot price and Bound para.
    :param spotPrice: underlying asset current price, not necessary when provided priceDf
    :param interval: how many standard deviation to calculate the expected price range
    :param Bound: underlying asset range bound to performe calculate on. eg. [50,100]
    :param strikePriceRange: strike price range, 0.3 stands for 30 percent above or below the current underlying asset price
    :param impiledVolRange: implied volatility range, 0.3 stands for 30 percents above or below the option's impiled volality
    :return:
        pre-compute data, strike price, asset's current price
    """

    daysTillExpir = (pd.to_datetime(expiryDate) - datetime.now())/pd.to_timedelta('1d')

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
            (optionDf['expiration'] == expiryDate) & (optionDf['K'] <= spotPrice * upperK) & (optionDf['K'] >= spotPrice * lowerK)
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
        d = list(joined1.groupby('is_call'))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                tempd = {}
                for key in ['buy_price', 'sell_price', 'spot_price', 'buy_delta', 'sell_delta', 'buy_vega', 'sell_vega',
                     'buy_gamma', 'sell_gamma', 'buy_theta', 'sell_theta', 'spot_per', 'per_ivs']:
                    tempd[key] = stri[1][key].values
                preOption1[type1[0]][stri[0]] = tempd


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
        d = list(joined1.groupby('is_call'))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                tempd = {}
                for key in ['buy_price', 'sell_price', 'spot_price', 'buy_delta', 'sell_delta', 'buy_vega', 'sell_vega',
                     'buy_gamma', 'sell_gamma', 'buy_theta', 'sell_theta', 'spot_per', 'per_ivs']:
                    tempd[key] = stri[1][key].values
                preOption2[type1[0]][stri[0]] = tempd


        preOption3 = {'C': {}, 'P': {}}
        d = list(option_data.groupby('is_call'))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                preOption3[type1[0]][stri[0]] = stri[1][
                    ['askIV', 'bidIV','expiry']].to_dict('list')

        return [preOption1,preOption2,preOption3],joined1['K'].unique(),spotPrice
    else:
        raise ValueError(" incorrect form of price bound")