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
    rolling_len = int(len(close))
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

def calculate_strategy_expiry_payoff(options, preOption2, map_bs):
    total_payoff = 0
    for option in options:
        total_payoff += preOption2[option['type']][option['strike']][map_bs[option['map']] + 'price'] * \
                        option['quantity']
    return total_payoff


def calculate_strategy_net_premium(options, spot_price):
    total_payoff = 0
    for option in options:
        total_payoff += option['quantity'] * calculate_price(option, spot_price)
    return total_payoff


def calculate_strategy_payoff(options, preOption, map_bs):
    total_payoff = 0
    for option in options:
        total_payoff += preOption[option['type']][option['strike']][map_bs[option['map']] + 'price'] * \
                        option['quantity']
    return total_payoff


def calculate_strategy_theta(options, preOption, map_bs):
    total_payoff = 0
    for option in options:
        total_payoff += preOption[option['type']][option['strike']][map_bs[option['map']] + 'theta'] * \
                        option['quantity']
    return total_payoff


def calculate_strategy_delta(options, preOption, map_bs):
    total_payoff = 0
    for option in options:
        total_payoff += preOption[option['type']][option['strike']][map_bs[option['map']] + 'delta'] * \
                        option['quantity']
    return total_payoff


def calculate_strategy_vega(options, preOption, map_bs):
    total_payoff = 0
    for option in options:
        total_payoff += preOption[option['type']][option['strike']][map_bs[option['map']] + 'vega'] * \
                        option[
                            'quantity']
    return total_payoff


def calculate_strategy_gamma(options, preOption, map_bs):
    total_payoff = 0
    for option in options:
        total_payoff += preOption[option['type']][option['strike']][map_bs[option['map']] + 'gamma'] * \
                        option['quantity']
    return total_payoff


def calculate_strategy_worstBestCase(min_index, max_index, options, preOption):
    option = options[0]
    per_ivs = preOption[option['type']][option['strike']]['per_ivs']
    spot_price = preOption[option['type']][option['strike']]['spot_price']
    worst_case_vol = per_ivs[min_index]
    worst_case_price = spot_price[min_index]
    best_case_vol = per_ivs[max_index]
    best_case_price = spot_price[max_index]
    return worst_case_vol, worst_case_price, best_case_vol, best_case_price


def calculate_strategy_premium(options, spot_price):
    total_payoff = 0
    for option in options:
        total_payoff += abs(option['quantity']) * calculate_price(option, spot_price)
    return total_payoff


def calculate_strategy_delta_starting_point(options, spot_price):
    total_payoff = 0
    for option in options:
        total_payoff += option['quantity'] * calculate_delta(option, spot_price)
    return total_payoff