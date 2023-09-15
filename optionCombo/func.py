import pandas as pd
import numpy as np
import math
import py_vollib.black_scholes as bs
import py_vollib.black_scholes.greeks.numerical as greeks
import py_vollib_vectorized # needed for computation
import itertools
import warnings
warnings.filterwarnings('ignore')

def gcd_many(s):
    s = np.abs(s)
    return np.gcd.reduce(s)

def is_two_dimensional(lst):
    if isinstance(lst, list):
        return all(isinstance(elem, list) for elem in lst)
    return False

def get_combinations(k_list, n):
    return [combo for combo in itertools.combinations(k_list, n) if all(combo[i] <= combo[i+1] for i in range(n-1))]

def is_symmetric_arithmetic_sequence(lst):
    
    if len(lst) <= 1:
        return True

    differences = [lst[i + 1] - lst[i] for i in range(len(lst) - 1)]
    n = len(differences)
    
    for i in range(n // 2):
        if differences[i] != differences[n - i - 1]:
            return False

    return True

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
    #delta = 0.0
    vega = 0.0
    payoff = 0.0
    for option in options:
        temp = option['pre1']
        map_value = map_bs[option['map']]
        quantities = option['quantity']
        theta += temp[map_value + 'theta'] * quantities
        #delta += temp[map_value + 'delta'] * quantities
        vega += temp[map_value + 'vega'] * quantities
        gamma += temp[map_value + 'gamma'] * quantities
        payoff += temp[map_value + 'price'] * quantities

    return vega, theta, gamma, payoff

def extend_bound_delta(options, map_bs):
    delta = 0.0

    for option in options:
        temp = option['pre5']
        map_value = map_bs[option['map']]
        quantities = option['quantity']
        delta += temp[map_value + 'delta'] * quantities
    return np.mean(delta),np.std(delta)

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
    return worst_case_vol, worst_case_price, best_case_vol, best_case_price, worst_case_days, best_case_days

def calculate_strategy_premium(options, map_bs):
    net_premium = 0
    premium = 0
    for option in options:
        quantity = option['quantity']
        map = map_bs[option['map']]
        premium += abs(quantity) * option['pre4'][map + 'price']
        net_premium += quantity * option['pre4'][map + 'price']
    return premium[0],net_premium[0]

def calculate_strategy_delta_starting_point(options, map_bs):
    total_delta = 0
    for option in options:
        map = map_bs[option['map']]
        total_delta += option['quantity'] * option['pre4'][map + 'delta']
    return total_delta[0]
