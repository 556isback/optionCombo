from concurrent.futures import ThreadPoolExecutor
import itertools
from optionCombo.func import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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

        '''if len(mn) == 1:   
            combos_2 = mnmb[2]
            for stri in combos_2:
                map_r = [1 if ab > 0 else -1 for ab in mb ]

                vols = float(df.loc[(df['K'] == stri[0]) & (df['option_type'] == map_cp[mn[0]])][map_ab[map_r[0]]].values/100) 
                if min(vols) > 0:
                    res.append(model_find_v2(spot_price, expir, stris, [vols], True, mn, mb))'''

        if len(mn) == 2:
            for stri in combos:
                stris = [stri[0], stri[1]]
                map_r = [1 if ab > 0 else -1 for ab in mb]
                vols = [float(self.preOption3[map_cp[m]][stri_temp][map_ab[b]][0]) for stri_temp, m, b in
                        zip(stris, mn, map_r)]
                expirys = [self.preOption3[map_cp[m]][stri_temp]['expiry'][0] for stri_temp, m, b in
                           zip(stris, mn, map_r)]
                if min(vols) > 0:
                    res.append([self.spot_price, expirys, stris, vols, mn, mb])

        if len(mn) == 3:
            for stri in combos:
                stris = [stri[0], stri[1], stri[2]]
                if stri[0] - stri[1] == stri[1] - stri[2] and stri[0] < self.spot_price and stri[2] > self.spot_price:
                    map_r = [1 if ab > 0 else -1 for ab in mb]

                    vols = [float(self.preOption3[map_cp[m]][stri_temp][map_ab[b]][0]) for stri_temp, m, b in
                            zip(stris, mn, map_r)]
                    expirys = [self.preOption3[map_cp[m]][stri_temp]['expiry'][0] for stri_temp, m, b in
                               zip(stris, mn, map_r)]
                    if min(vols) > 0:
                        res.append([self.spot_price, expirys, stris, vols, mn, mb])

        if len(mn) == 4:
            for stri in combos:
                if stri[0] - stri[1] == stri[2] - stri[3] and stri[0] != stri[1] and (
                        stri[1] < self.spot_price and stri[2] > self.spot_price):

                    stris = [stri[0], stri[1], stri[2], stri[3]]
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

        model_n = []
        model_b = []
        temp = []
        optiontypes = self.optionType
        tradetypes = self.tradeTypes
        maxquantity = 4
        quantityRange = np.array([i + 1 for i in range(maxquantity)])
        quantityRange = np.append(quantityRange, quantityRange * -1)
        quantity = quantityRange
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
            unique_k = self.strikePrice

            combos_2 = [(stri_1, stri_2) for stri_1 in unique_k
                        for stri_2 in unique_k
                        if np.all(np.diff([stri_1, stri_2]) >= 0)]
            combos_3 = [(stri_1, stri_2, stri_3) for stri_1 in unique_k
                        for stri_2 in unique_k
                        for stri_3 in unique_k
                        if np.all(np.diff([stri_1, stri_2, stri_3]) >= 0)]
            combos_4 = [(stri_1, stri_2, stri_3, stri_4) for stri_1 in unique_k
                        for stri_2 in unique_k
                        for stri_3 in unique_k
                        for stri_4 in unique_k
                        if np.all(np.diff([stri_1, stri_2, stri_3, stri_4]) >= 0)]

            loops = []
            for mn, mb in zip(model_n, model_b):
                if len(mn) == 1:
                    pass
                    # loops.append([mn, mb, unique_k.reshape(-1, 1)])
                elif len(mn) == 2:
                    loops.append([mn, mb, combos_2])
                elif len(mn) == 3:
                    loops.append([mn, mb, combos_3])
                elif len(mn) == 4:
                    loops.append([mn, mb, combos_4])

            pool = ThreadPoolExecutor(max_workers=8)
            paras = []
            for asset in tqdm(pool.map(self.pool_func, loops), total=len(loops)):
                paras.extend(asset)

            pool = ThreadPoolExecutor(max_workers=16)
            res = []
            for asset in tqdm(pool.map(self.model_find_v2, paras), total=len(paras)):
                res.append(asset)

            df = pd.DataFrame(res, columns=['para', 'stra', 'maxRisk', 'probal', 'RR', 'wv', 'wp', 'bv', 'bp','wd','bd',
                                            'delta_starting',
                                            'mean_delta', 'std_delta', 'mean_vega', 'mean_theta', 'std_theta',
                                            'premium']).dropna()
            return df

    def model_find_v2(self, paras):
        """
        this is a function call by options_model_finder after geeting all the combinations, it calculate the strategy's geeks, and all kinds of stats, and return the result
        :param paras:
        :return:
        """

        spot_price, expirys, strs, vols, model, model_b = paras

        preOption, preOption2 = self.preOption, self.preOption2
        quty = [1] * len(strs)
        map_type = {1: 'C', -1: 'P'}
        types = [map_type[i] for i in model]
        quty = [i * j for i, j in zip(quty, model_b)]
        map_fu = [1 if qu > 0 else -1 for qu in model_b]
        map_bs = {1: 'buy_', -1: 'sell_'}
        # Define call and put options
        options = [{'type': type1, 'strike': str1, 'expiry': expiry, 'vol': vol, 'quantity': quty_1, 'map': map_1,'pre1':preOption[type1][str1],'pre2':preOption2[type1][str1]}
                   for type1, str1, vol, quty_1, map_1, expiry in zip(types, strs, vols, quty, map_fu, expirys)]

        payoffs = calculate_strategy_expiry_payoff(options, map_bs)
        premium, net_premium = calculate_strategy_premium(options, spot_price)
        deltas, vegas, thetas, _, allTimePayoffs = calculate_strategy_stats(options, map_bs)
        profit = len(payoffs[payoffs > net_premium])
        loss = len(payoffs[payoffs < net_premium])
        min_payoffs = min(payoffs)
        probal = profit / (profit + loss)
        RR = abs(max(payoffs) - net_premium) / abs(net_premium - min(payoffs))
        allTimePayoffs = list(allTimePayoffs)
        min_index = allTimePayoffs.index(min(allTimePayoffs))
        max_index = allTimePayoffs.index(max(allTimePayoffs))
        wv, wp, bv, bp, wd, bd = calculate_strategy_worstBestCase(min_index, max_index, options)
        delta_starting_point = calculate_strategy_delta_starting_point(options, spot_price).values[0][0]
        straSym = '__'.join(
            [map_type[cp] + '_' + str(int(k)) + '_' + str(quty) for k, cp, quty in zip(strs, model, model_b)])

        return (
            [spot_price, expirys, strs, vols, model, model_b], straSym,
            abs(min_payoffs - net_premium) / abs(net_premium), probal, RR,
            wv, wp, bv, bp, wd, bd, abs(delta_starting_point), np.mean([abs(i) for i in deltas]), np.std(deltas),
            np.mean(vegas),
            np.mean(thetas), np.std(thetas), premium)

    def model_plot(self, paras):

        """
        this is a function to visulazing the result, bule line is payoff curve, red line stands for net premium
        :param paras: a colume from the output DataFrame of options_model_finder
        :return:
        """

        spot_price, expirys, strs, vols, model, model_b = paras
        temp = spot_price
        risk_free_rate = 0.02
        preOption, preOption2 = self.preOption, self.preOption2
        quty = [1] * len(strs)
        map_type = {1: 'C', -1: 'P'}
        types = [map_type[i] for i in model]
        quty = [i * j for i, j in zip(quty, model_b)]
        map_fu = [1 if qu > 0 else -1 for qu in model_b]
        map_bs = {1: 'buy_', -1: 'sell_'}
        map_ba = {1: 'bid_ivs', -1: 'ask_ivs'}
        # Define call and put options
        options = [{'type': type1, 'strike': str1, 'expiry': expiry, 'vol': vol, 'quantity': quty_1, 'map': map_1,'pre1':preOption[type1][str1],'pre2':preOption2[type1][str1]}
                   for type1, str1, vol, quty_1, map_1, expiry in zip(types, strs, vols, quty, map_fu, expirys)]

        payoffs = calculate_strategy_expiry_payoff(options, map_bs)
        premium,net_premium = calculate_strategy_premium(options, spot_price)
        deltas,vegas,thetas,_,allTimePayoffs = calculate_strategy_stats(options, map_bs)
        profit = len(payoffs[payoffs > net_premium])
        loss = len(payoffs[payoffs < net_premium])
        min_payoffs = min(payoffs)
        probal = profit / (profit + loss)
        RR = abs(max(payoffs) - net_premium) / abs(net_premium - min(payoffs))
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
        print('max risk ' + str(abs(min_payoffs - net_premium) / [abs(net_premium)]))
        print('worst case, vol: ' + str(wv) + ' price: ' + str(wp)+ ' days: ' + str(wd))
        print('best case, vol: ' + str(bv) + ' price: ' + str(bp)+ ' days: ' + str(bd))
        print('min theta ' + str(min(thetas)))
        print('min vega ' + str(min(vegas)))
        print(straSym)
        # print(temp, time_to_expiry, strs, vols, buy, model, model_b)

