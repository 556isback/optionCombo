import pandas as pd
import numpy as np
import math
import py_vollib.black_scholes as bs
import py_vollib.black_scholes.greeks.numerical as greeks
import py_vollib_vectorized # needed for computation



def gcd_many(s):
    s = [abs(j) for j in s]
    g = 0
    for i in range(len(s)):
        if i == 0:
            g = s[i]
        else:
            g=math.gcd(g,s[i])

    return g

def is_two_dimensional(lst):
    if isinstance(lst, list):
        return all(isinstance(elem, list) for elem in lst)
    return False

def has_same_form(list1, list2):
    if len(list1) != len(list2):
        return False

    for item1, item2 in zip(list1, list2):
        if type(item1) != type(item2):
            return False
        if isinstance(item1, list) and isinstance(item2, list):
            if not has_same_form(item1, item2):
                return False
    return True


def all_positive(l1,l2):
    numbers = [a*b for a,b in zip(l1,l2)]
    for num in numbers:
        if num <= 0:
            return False
    return True

def correct_form(list1, list2):
    if len(list1) != len(list2):
        return False
    for i,j in zip(list1,list2):
        if is_two_dimensional(j):
            for ele in j:
                if not has_same_form(i,ele):
                    return False
        else:
            if not has_same_form(i,j):
                return False
    return True

def calRange(df,daysTillExpir,interval_width = 2):
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



def calculate_price(option, spot_price_1,risk_free_rate = 0.02):
    return bs.black_scholes(option['type'].lower(), spot_price_1, option['strike'], option['expiry'],
                            risk_free_rate,
                            option['vol'])


def calculate_delta(option, spot_price_1,risk_free_rate = 0.02):
    return greeks.delta(option['type'].lower(), spot_price_1, option['strike'], option['expiry'],
                        risk_free_rate,
                        option['vol'])


def calculate_strategy_expiry_payoff(options, map_bs):
    total_payoff = 0
    for option in options:
        total_payoff += option['pre2'][map_bs[option['map']] + 'price'] * \
                        option['quantity']
    return total_payoff


def calculate_strategy_stats(options, map_bs):
    theta = 0.0
    gamma = 0.0
    delta = 0.0
    vega = 0.0
    payoff = 0.0

    for option in options:
        temp = option['pre1']
        map_value = map_bs[option['map']]
        quantities = option['quantity']
        theta += temp[map_value + 'theta'] * quantities
        delta += temp[map_value + 'delta'] * quantities
        vega += temp[map_value + 'vega'] * quantities
        gamma += temp[map_value + 'gamma'] * quantities
        payoff += temp[map_value + 'price'] * quantities

    return delta, vega, theta, gamma, payoff

def calculate_strategy_worstBestCase(min_index, max_index, options):
    option = options[0]['pre1']
    per_ivs = option['per_ivs']
    spot_price = option['spot_price']
    days = option['expirys']
    worst_case_vol = per_ivs[min_index]
    worst_case_price = spot_price[min_index]
    worst_case_days = days[min_index]*365
    best_case_vol = per_ivs[max_index]
    best_case_price = spot_price[max_index]
    best_case_days = days[max_index]*365
    return worst_case_vol, worst_case_price, best_case_vol, best_case_price,worst_case_days,best_case_days

def calculate_strategy_premium(options, spot_price):
    net_premium = 0
    premium = 0
    for option in options:
        quantity = option['quantity']
        net_premium += abs(quantity) * calculate_price(option, spot_price)
        premium += quantity * calculate_price(option, spot_price)
    return premium.values[0][0],net_premium.values[0][0]

def calculate_strategy_delta_starting_point(options, spot_price):
    total_payoff = 0
    for option in options:
        total_payoff += option['quantity'] * calculate_delta(option, spot_price)
    return total_payoff
