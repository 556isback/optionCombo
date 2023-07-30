from concurrent.futures import ThreadPoolExecutor
import itertools
from optionCombo.func import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class option_model:
    def __init__(self, spot_price, strikePrice, preoption, optiontypes = [[1, 1, 1, 1]], tradetypes = None, maxquantity=3):

        """
        this is a class to performe calculation and plotting of option strategy
        :param spot_price: underlying asset's price, out put of preInit.Prep
        :param strikePrice:  strike price range, out put of preInit.Prep
        :param preoption: pre-compute option data, out put of preInit.Prep
        :param optiontypes: options type to performe calculations on, eg. [[1,1,1,1],[-1,-1]], it calculate the possible combinations of four call options and two put options. 1 stands for call option, -1 stands for put options
        :param tradetypes: sell or buy options correspondingly, None by default to include all combinations, eg. if optiontypes =[[1,1,1]], tradetypes should be [[1,-1,1]] or [[[1,-1,1],[1,1,1]]]
        :param maxquantity: maxquantity is the input for the maximum potential quantity of a single option trade amount
        """

        self.spot_price = spot_price
        self.strikePrice = strikePrice
        array = np.array(optiontypes, dtype=object)
        shape = len(array.shape)
        if not is_two_dimensional(optiontypes) or not optiontypes:
            raise ValueError('incorrect form, eg. [[1,1,1]] or [[1,1,1],[1,-1,1]] when opt for multiple types of stra')
        for arr in optiontypes:
            if len(arr) < 5 and len(arr) > 1:
                if sum(map(abs, arr)) == len(arr):
                    pass
                else:
                    raise ValueError(
                        'ignore ' + str(arr) + ', use 1 or -1 to represent call or put options, eg. [[1,1,1]]')
            else:
                raise ValueError('ignore ' + str(
                    arr) + " ,currently only accept option combo's length range from 2 to 4, as these is what's normal for options stras, eg. [[1,1,1,1],[-1,-1,-1]] ")
        if tradetypes:
            if not correct_form(optiontypes, tradetypes):
                raise ValueError(
                    'ignore ' + str(
                        tradetypes) + " , incorrect form of tradetypes, eg. if optiontypes =[[1,1,1]], tradetypes should be [[1,-1,1]] or [[[1,-1,1],[1,1,1]]] ")
        self.optionType = optiontypes
        self.tradeTypes = tradetypes
        self.preOption = preoption[0]
        self.preOption2 = preoption[1]
        self.preOption3 = preoption[2]
        self.preOption4 = preoption[3]
        self.preOption5 = preoption[4]
        quantityRange = np.array([i + 1 for i in range(maxquantity)])
        quantityRange = np.append(quantityRange, quantityRange * -1)
        self.quantity = quantityRange

    def pool_func(self, mnmb):

        """
        function called by options_model_finder for getting the combinations of strike price and given optionTypes parameter from Init function,
        it exclude the result eg. when the bid iv is 0 which is impossible to write the option, also it exculde the result of strike price too far away from current underlying asset's price
        :param mnmb: mn an instance of optionType eg [1,1,1], mb trade quantity eg. [-1,2,1]
        :return: combinations of option strategy
        """

        mn, mb, combos = mnmb
        map_cp = {1: 'C', -1: 'P'}
        map_ab = {1: 'askIV', -1: 'bidIV'}
        res = []

        if len(mn) == 2:
            for stris in combos:
                map_r = [1 if ab > 0 else -1 for ab in mb]
                vols = [float(self.preOption3[map_cp[m]][stri_temp][map_ab[b]][0]) for stri_temp, m, b in
                        zip(stris, mn, map_r)]
                expirys = [self.preOption3[map_cp[m]][stri_temp]['expiry'][0] for stri_temp, m, b in
                           zip(stris, mn, map_r)]
                if min(vols) > 0:
                    res.append([self.spot_price, expirys, stris, vols, mn, mb])

        if len(mn) > 2:
            for stris in combos:
                if is_symmetric_arithmetic_sequence(stris):
                    map_r = [1 if ab > 0 else -1 for ab in mb]

                    vols = [float(self.preOption3[map_cp[m]][stri_temp][map_ab[b]][0]) for stri_temp, m, b in
                            zip(stris, mn, map_r)]
                    expirys = [self.preOption3[map_cp[m]][stri_temp]['expiry'][0] for stri_temp, m, b in
                               zip(stris, mn, map_r)]
                    if min(vols) > 0:
                        res.append([self.spot_price, expirys, stris, vols, mn, mb])


        return res

    def options_model_finder(self):

        """
        :return: return the DataFrame containing strategy's stats
        """

        temp = []
        optiontypes = self.optionType
        tradetypes = self.tradeTypes
        quantity = self.quantity

        if tradetypes:
            for arr, arr1 in zip(optiontypes, tradetypes):
                if not is_two_dimensional(arr1):
                    arr1 = [arr1]
                k = quantity
                # generate all possible combinations of the integers
                combinations = itertools.product(k, repeat=len(arr))
                for com in combinations:
                    for form in arr1:
                        if all_positive(com, form) > 0:
                            cDivisor = gcd_many(com)
                            tempRe = tuple(np.array(com) / cDivisor)
                            repeat = [arr, tempRe] in temp
                            if not repeat:
                                temp.append([arr, com])
        else:
            for arr in optiontypes:
                k = quantity
                # generate all possible combinations of the integers
                combinations = itertools.product(k, repeat=len(arr))
                for com in combinations:
                    cDivisor = gcd_many(com)
                    tempRe = tuple(np.array(com) / cDivisor)
                    repeat = [arr, tempRe] in temp
                    if not repeat:
                        temp.append([arr, com])
        model_n, model_b = zip(*temp)

        if model_n and model_b:
            len_n = list(set(list(map(len, model_n))))
            unique_k = self.strikePrice

            combos = {n:get_combinations(unique_k, n) for n in len_n }

            loops = []
            for mn, mb in zip(model_n, model_b):
                loops.append([mn, mb, combos[len(mn)]])

            #pool = ThreadPoolExecutor(max_workers=8)
            paras = []
            for loop in tqdm(loops):
                paras.extend(self.pool_func(loop))
            #pool.shutdown()
            '''pool = ThreadPoolExecutor(max_workers=4)
            res = []
            for asset in tqdm(pool.map(self.model_find_v2, paras), total=len(paras)):
                res.append(asset)
            pool.shutdown()'''
            res = []
            for para in tqdm(paras):
                res.append(self.model_find_v2(para))
            df = pd.DataFrame(res, columns=['para', 'stra', 'maxRisk', 'probal', 'RR', 'wv', 'wp', 'bv', 'bp', 'wd','bd',
                                            'mean_delta', 'std_delta', 'mean_vega', 'mean_theta', 'std_theta',
                                            'premium','minReward']).dropna()

            return df

    def model_find_v2(self, paras):
        """
        this is a function call by options_model_finder after geeting all the combinations, it calculate the strategy's geeks, and all kinds of stats, and return the result
        :param paras:
        :return:
        """

        spot_price, expirys, strs, vols, model, model_b = paras

        preOption, preOption2,preOption4,preOption5 = self.preOption, self.preOption2,self.preOption4,self.preOption5
        quty = [1] * len(strs)
        map_type = {1: 'C', -1: 'P'}
        types = [map_type[i] for i in model]
        quty = [i * j for i, j in zip(quty, model_b)]
        map_fu = [1 if qu > 0 else -1 for qu in model_b]
        map_bs = {1: 'buy_', -1: 'sell_'}
        # Define call and put options
        options = [{'type': type1, 'strike': str1, 'expiry': expiry, 'vol': vol, 'quantity': quty_1, 'map': map_1,
                    'pre1': preOption[type1][str1], 'pre2': preOption2[type1][str1],
                    'pre4': preOption4[type1][str1],'pre5':preOption5[type1][str1]} for type1, str1, vol, quty_1, map_1, expiry in
                   zip(types, strs, vols, quty, map_fu, expirys)]

        payoffs = calculate_strategy_expiry_payoff(options, map_bs)
        premium, net_premium = calculate_strategy_premium(options, map_bs)
        vegas, thetas, _, allTimePayoffs = calculate_strategy_stats(options, map_bs)
        mean_delta,std_delta = extend_bound_delta(options, map_bs)
        profit = len(payoffs[payoffs > net_premium])
        loss = len(payoffs[payoffs < net_premium])
        min_payoffs = min(allTimePayoffs)
        try:
            probal = profit / (profit + loss)
        except ZeroDivisionError:
            probal = 0

        RR = abs(max(payoffs) - net_premium) / abs(net_premium - min_payoffs)
        minReward = min(payoffs) - net_premium
        allTimePayoffs = list(allTimePayoffs)
        min_index = allTimePayoffs.index(min(allTimePayoffs))
        max_index = allTimePayoffs.index(max(allTimePayoffs))
        wv, wp, bv, bp, wd, bd = calculate_strategy_worstBestCase(min_index, max_index, options)
        #delta_starting_point = calculate_strategy_delta_starting_point(options, map_bs)
        straSym = '__'.join(
            [map_type[cp] + '_' + str(int(k)) + '_' + str(quty) for k, cp, quty in zip(strs, model, model_b)])

        return (
            [spot_price, expirys, strs, vols, model, model_b], straSym,
            abs(min_payoffs - net_premium) / abs(net_premium), probal, RR,wv, wp, bv, bp,  wd, bd,
             mean_delta , std_delta,
            np.mean(vegas),
            np.mean(thetas), np.std(thetas), premium,minReward)

    def model_plot(self, paras):

        """
        this is a function to visulazing the result, bule line is payoff curve, red line stands for net premium
        :param paras: a colume from the output DataFrame of options_model_finder
        :return:
        """

        spot_price, expirys, strs, vols, model, model_b = paras
        preOption, preOption2, preOption4, preOption5 = self.preOption, self.preOption2, self.preOption4, self.preOption5
        quty = [1] * len(strs)
        map_type = {1: 'C', -1: 'P'}
        types = [map_type[i] for i in model]
        quty = [i * j for i, j in zip(quty, model_b)]
        map_fu = [1 if qu > 0 else -1 for qu in model_b]
        map_bs = {1: 'buy_', -1: 'sell_'}
        # Define call and put options
        options = [{'type': type1, 'strike': str1, 'expiry': expiry, 'vol': vol, 'quantity': quty_1, 'map': map_1,
                    'pre1': preOption[type1][str1], 'pre2': preOption2[type1][str1],
                    'pre4': preOption4[type1][str1], 'pre5': preOption5[type1][str1]} for
                   type1, str1, vol, quty_1, map_1, expiry in
                   zip(types, strs, vols, quty, map_fu, expirys)]

        payoffs = calculate_strategy_expiry_payoff(options, map_bs)
        premium, net_premium = calculate_strategy_premium(options, map_bs)
        vegas, thetas, _, allTimePayoffs = calculate_strategy_stats(options, map_bs)
        mean_delta, std_delta = extend_bound_delta(options, map_bs)
        profit = len(payoffs[payoffs > net_premium])
        loss = len(payoffs[payoffs < net_premium])
        min_payoffs = min(allTimePayoffs)
        probal = profit / (profit + loss)
        RR = abs(max(payoffs) - net_premium) / abs(net_premium - min_payoffs)
        allTimePayoffs = list(allTimePayoffs)
        min_index = allTimePayoffs.index(min(allTimePayoffs))
        max_index = allTimePayoffs.index(max(allTimePayoffs))
        wv, wp, bv, bp, wd, bd = calculate_strategy_worstBestCase(min_index, max_index, options)
        # Plot strategy metrics

        #per_ivs = preOption2[options[0]['type']][options[0]['strike']]['per_ivs']
        spot_per = preOption2[options[0]['type']][options[0]['strike']]['spot_price']

        ax = sns.lineplot(x=spot_per, y=payoffs)
        ax.axhline(net_premium, color='red')

        plt.xlabel('Spot Price')
        plt.ylabel('payoff')
        plt.show()
        straSym = '__'.join(
            [map_type[cp] + '_' + str(int(k)) + '_' + str(quty) for k, cp, quty in zip(strs, model, model_b)])
        # print(prices)
        print('net premium ' + str(net_premium))
        print('premium ' + str(premium))
        print('Risk Reward ' + str(RR))
        print('probal ' + str(probal))
        print('lowest possible premium ' + str(min_payoffs))
        print('max risk ' + str((abs(min_payoffs - net_premium) / [abs(net_premium)])[0]))
        print('worst case, vol: ' + str(wv) + ' price: ' + str(wp) +' daysTillExpir: ' + str(wd))
        print('best case, vol: ' + str(bv) + ' price: ' + str(bp) +' daysTillExpir: ' + str(bd))
        print('mean delta' + str(mean_delta))
        print('min theta ' + str(min(thetas)))
        print('min vega ' + str(min(vegas)))
        print(straSym)
        # print(temp, time_to_expiry, strs, vols, buy, model, model_b)

