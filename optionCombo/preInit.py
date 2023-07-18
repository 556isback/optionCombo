import pandas as pd
from optionCombo.func import *
import pandas as pd
import py_vollib.black_scholes as bs
import py_vollib.black_scholes.greeks.numerical as greeks
import py_vollib_vectorized
from datetime import datetime , timezone, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calRange(df,expiryDate,interval_width = 2):
    """

    :param df: DataFrame containing underlying asset's price data, it is used for determine expected price range and get current asset price
    :param expiryDate: expiry date of the option you want to trade on, doesn't support multiple dates at the moment
    :param interval_width: how many standard deviation to calculate the expected price range
    :return: lowerBound, upperBound

    """
    daysTillExpir = (pd.to_datetime(expiryDate) - datetime.now()) / pd.to_timedelta('1d')
    close   = df['close']
    minutes = df.Timestamp.diff().values[-1] / np.timedelta64(60, 's')
    forward = daysTillExpir
    theta   = 60 * 24 * forward / minutes
    rolling_len = int(len(close)/4*3)
    # Expected Range Model
    df['return'] = close.pct_change().dropna()
    log_return = df['return']
    average = log_return.rolling(rolling_len).mean().values[-1]
    stdev = log_return.rolling(rolling_len).std().values[-1]
    theta_average = theta * average
    theta_stdev = stdev * math.sqrt(theta)
    upper0 = close * math.exp(theta_average + interval_width * theta_stdev)
    lower0 = close  * math.exp(theta_average - interval_width * theta_stdev)
    lower0,upper0 = lower0.values[-1],upper0.values[-1]

    return lower0,upper0

def Prep(expiryDate, optionDf, spotPrice = None, Bound = None, BoundExtend = 1.2 ,strikePriceRange = 0.3 , impiledVolRange = 0.3 ):

    """
    this function is for pre-computation, like option delta, vega, theta, and option price.
    it calculate the above stats in a preset range eg. impiledVolRange, price range.

    :param expiryDate: expiry date of the option you want to trade on, doesn't support multiple dates at the moment
    :param optionDf: DataFrame containing option's expiry date, bid ask Iv, strike price, and option type
    :param spotPrice: underlying asset current price, not necessary when provided priceDf
    :param Bound: underlying asset range bound to performe calculate on. eg. [50,100]
    :param BoundExtend: underlying asset range bound to performe calculation on Delta stats, useful when trying to find delta neutral strategy. eg. 1.2
    :param strikePriceRange: strike price range, 0.3 stands for 30 percent above or below the current underlying asset price
    :param impiledVolRange: implied volatility range, 0.3 stands for 30 percents above or below the option's impiled volality
    :return:pre-compute data, strike price, asset's current price
    """


    lowerB, upperB = None,None


    if len(Bound) == 2:
        lowerB,upperB = min(Bound),max(Bound)

    if lowerB and upperB:
        lowerK,upperK = 1-strikePriceRange/2,1+strikePriceRange/2

        lowerIV,upperIV = 1-impiledVolRange/2,1+impiledVolRange/2



        option_data = optionDf.loc[
            (optionDf['expiration'] == expiryDate) & (optionDf['K'] <= spotPrice * upperK) & (optionDf['K'] >= spotPrice * lowerK)
        ]
        tempD = option_data

        option_data['expirys'] = option_data['expiry'].apply(lambda x: list(np.linspace(0, x, 2)), 0)
        option_data = option_data.explode(['expirys']).reset_index(drop=True)

        temp = pd.DataFrame()
        option_data['per_ivs'] = option_data['bidIV'].apply(lambda x: list(np.linspace(1 * lowerIV, 1 * upperIV, 2)),0)
        option_data['askIVs'] = option_data['askIV'].apply(lambda x: list(np.linspace(x * lowerIV, x * upperIV, 2)), 0)
        option_data['bidIVs'] = option_data['bidIV'].apply(lambda x: list(np.linspace(x * lowerIV, x * upperIV, 2)), 0)
        temp['askIVs'] = option_data.explode(['askIVs']).reset_index(drop=True)['askIVs']
        temp['per_ivs'] = option_data.explode(['per_ivs']).reset_index(drop=True)['per_ivs']
        temp['bidIVs'] = option_data.explode(['bidIVs']).reset_index(drop=True)['bidIVs']
        option_data = option_data.drop(['askIVs','per_ivs','bidIVs'],axis=1)

        option_data = option_data.join(temp, how='cross')

        joined1 = option_data

        spotPrices = np.linspace(lowerB, upperB, 30)
        spotPer = np.linspace(1 * lowerB / spotPrice, 1 * upperB / spotPrice, 30)
        spotPrices = pd.DataFrame({'spot_price': spotPrices, 'spot_per': spotPer})

        joined1 = joined1.join(spotPrices, how='cross')

        risk_free_rate = 0.02

        joined1['buy_price'] = \
        bs.black_scholes(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                         joined1['askIVs']).values.reshape(1, -1)[0]
        joined1['sell_price'] = \
        bs.black_scholes(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                         joined1['bidIVs']).values.reshape(1, -1)[0]
        joined1['buy_delta'] = \
        greeks.delta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                     joined1['askIVs']).values.reshape(1, -1)[0]
        joined1['sell_delta'] = \
        greeks.delta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                     joined1['bidIVs']).values.reshape(1, -1)[0]
        joined1['buy_vega'] = \
        greeks.vega(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                    joined1['askIVs']).values.reshape(1, -1)[0]
        joined1['sell_vega'] = \
        greeks.vega(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                    joined1['bidIVs']).values.reshape(1, -1)[0]
        joined1['buy_gamma'] = \
        greeks.gamma(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                     joined1['askIVs']).values.reshape(1, -1)[0]
        joined1['sell_gamma'] = \
        greeks.gamma(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                     joined1['bidIVs']).values.reshape(1, -1)[0]
        joined1['buy_theta'] = \
        greeks.theta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                     joined1['askIVs']).values.reshape(1, -1)[0]
        joined1['sell_theta'] = \
        greeks.theta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'], risk_free_rate,
                     joined1['bidIVs']).values.reshape(1, -1)[0]

        joined1 = joined1.round(4)

        preOption1 = {'C': {}, 'P': {}}
        d = list(joined1.groupby('is_call'))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                tempd = {}
                for key in ['buy_price', 'sell_price', 'spot_price', 'buy_delta', 'sell_delta', 'buy_vega', 'sell_vega',
                     'buy_gamma', 'sell_gamma', 'buy_theta', 'sell_theta', 'spot_per','per_ivs','expirys']:
                    tempd[key] = stri[1][key].values
                preOption1[type1[0]][stri[0]] = tempd


        joined1 = tempD
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

        joined1 = joined1.round(4)

        preOption2 = {'C': {}, 'P': {}}
        d = list(joined1.groupby('is_call'))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                tempd = {}
                for key in ['buy_price', 'sell_price', 'spot_price', 'buy_delta', 'sell_delta', 'buy_vega', 'sell_vega',
                     'buy_gamma', 'sell_gamma', 'buy_theta', 'sell_theta', 'spot_per']:
                    tempd[key] = stri[1][key].values
                preOption2[type1[0]][stri[0]] = tempd


        preOption3 = {'C': {}, 'P': {}}
        d = list(option_data.groupby('is_call'))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                preOption3[type1[0]][stri[0]] = stri[1][
                    ['askIV', 'bidIV','expiry']].to_dict('list')
        spotPricecal = [spotPrice]

        joined1 = tempD
        #joined1 = joined1.join(spotPrices, how='cross')

        # expirys = pd.DataFrame({'expirys':expirys})
        # joined1 = joined1.join(expirys,how='cross')
        risk_free_rate = 0.02

        joined1['buy_price'] = \
            bs.black_scholes(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'],
                             risk_free_rate,
                             joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_price'] = \
            bs.black_scholes(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'],
                             risk_free_rate,
                             joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_delta'] = \
            greeks.delta(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'], risk_free_rate,
                         joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_delta'] = \
            greeks.delta(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'], risk_free_rate,
                         joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_vega'] = \
            greeks.vega(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'], risk_free_rate,
                        joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_vega'] = \
            greeks.vega(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'], risk_free_rate,
                        joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_gamma'] = \
            greeks.gamma(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'], risk_free_rate,
                         joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_gamma'] = \
            greeks.gamma(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'], risk_free_rate,
                         joined1['bidIV']).values.reshape(1, -1)[0]
        joined1['buy_theta'] = \
            greeks.theta(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'], risk_free_rate,
                         joined1['askIV']).values.reshape(1, -1)[0]
        joined1['sell_theta'] = \
            greeks.theta(joined1['is_call'].str.lower(), spotPricecal, joined1['K'], joined1['expiry'], risk_free_rate,
                         joined1['bidIV']).values.reshape(1, -1)[0]

        joined1 = joined1.round(4)

        preOption4 = {'C': {}, 'P': {}}
        d = list(joined1.groupby('is_call'))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                tempd = {}
                for key in ['buy_price', 'sell_price', 'buy_delta', 'sell_delta', 'buy_vega', 'sell_vega',
                            'buy_gamma', 'sell_gamma', 'buy_theta', 'sell_theta']:
                    tempd[key] = stri[1][key].values
                preOption4[type1[0]][stri[0]] = tempd

        joined1 = option_data

        spotPrices = np.linspace(lowerB*(2-BoundExtend), upperB*BoundExtend, 40)
        spotPer = np.linspace(1 * lowerB / spotPrice, 1 * upperB / spotPrice, 40)
        spotPrices = pd.DataFrame({'spot_price': spotPrices, 'spot_per': spotPer})

        joined1 = joined1.join(spotPrices, how='cross')

        risk_free_rate = 0.02

        joined1['buy_delta'] = \
            greeks.delta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'],
                         risk_free_rate,
                         joined1['askIVs']).values.reshape(1, -1)[0]
        joined1['sell_delta'] = \
            greeks.delta(joined1['is_call'].str.lower(), joined1['spot_price'], joined1['K'], joined1['expirys'],
                         risk_free_rate,
                         joined1['bidIVs']).values.reshape(1, -1)[0]


        joined1 = joined1.round(4)

        preOption5 = {'C': {}, 'P': {}}
        d = list(joined1.groupby('is_call'))
        for type1 in d:
            f = list(type1[1].groupby('K'))
            for stri in f:
                tempd = {}
                for key in ['buy_delta', 'sell_delta']:
                    tempd[key] = stri[1][key].values
                preOption5[type1[0]][stri[0]] = tempd

        return [preOption1,preOption2,preOption3,preOption4,preOption5],joined1['K'].unique(),spotPrice
    else:
        raise ValueError(" incorrect form of price bound")