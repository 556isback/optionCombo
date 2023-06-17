from concurrent.futures import ThreadPoolExecutor
import itertools
from optionCombo.func import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class option_model:
    def __init__(self, spot_price, df, preoption, optionstypes =[[1,1,1,1]], maxquantity = 3):
        self.spot_price = spot_price
        self.df = df
        array = np.array(optionstypes)
        shape = len(array.shape)
        if not is_two_dimensional(optionstypes) or not optionstypes:
            raise ValueError('incorrect form, eg. [[1,1,1]] or [[1,1,1],[1,-1,1]] when opt for multiple types of stra')
        for arr in optionstypes:
                if len(arr) < 5 and len(arr) > 1:
                    if sum(map(abs,arr)) == len(arr):
                        pass
                    else:
                        raise ValueError('ignore '+ str(arr)+', use 1 or -1 to represent call or put options, eg. [[1,1,1]]' )
                else:
                    raise ValueError('ignore '+ str(arr)+" ,currently only accept option combo's length range from 2 to 4, as these is what's normal for options stras, eg. [[1,1,1,1],[-1,-1,-1]] ")
        self.optionsType = optionstypes
        self.preOption = preoption[0]
        self.preOption2 = preoption[1]
        self.preOption3 = preoption[2]

        quantityRange = np.array([i + 1 for i in range(maxquantity)])
        quantityRange = np.append(quantityRange, quantityRange * -1)
        self.quantity = quantityRange

    def pool_func(self,mnmb):
        mn, mb = mnmb[0], mnmb[1]
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
            combos_2 = mnmb[2]
            for stri in combos_2:
                stris = [stri[0], stri[1]]
                map_r = [1 if ab > 0 else -1 for ab in mb]
                vols = [float(self.preOption3[map_cp[m]][stri_temp][map_ab[b]].values[0]) for stri_temp, m, b in
                        zip(stris, mn, map_r)]
                expirys = [self.preOption3[map_cp[m]][stri_temp]['expiry'].values[0] for stri_temp, m, b in
                           zip(stris, mn, map_r)]
                if min(vols) > 0:
                    res.append([self.spot_price, expirys, stris, vols, mn, mb])
        if len(mn) == 3:
            combos_3 = mnmb[2]
            for stri in combos_3:
                stris = [stri[0], stri[1], stri[2]]

                if stri[0] - stri[1] == stri[1] - stri[2] and stri[0] < self.spot_price and stri[2] > self.spot_price:
                    map_r = [1 if ab > 0 else -1 for ab in mb]

                    vols = [float(self.preOption3[map_cp[m]][stri_temp][map_ab[b]].values[0]) for stri_temp, m, b in
                            zip(stris, mn, map_r)]
                    expirys = [self.preOption3[map_cp[m]][stri_temp]['expiry'].values[0] for stri_temp, m, b in
                            zip(stris, mn, map_r)]
                    if min(vols) > 0:
                        res.append([self.spot_price, expirys, stris, vols, mn, mb])
        if len(mn) == 4:
            combos_4 = mnmb[2]
            for stri in combos_4:
                if stri[0] - stri[1] == stri[2] - stri[3] and stri[0] != stri[1] and (
                        stri[1] < self.spot_price and stri[2] > self.spot_price):

                    stris = [stri[0], stri[1], stri[2], stri[3]]
                    map_r = [1 if ab > 0 else -1 for ab in mb]

                    vols = [float(self.preOption3[map_cp[m]][stri_temp][map_ab[b]].values[0]) for stri_temp, m, b in
                            zip(stris, mn, map_r)]
                    expirys = [self.preOption3[map_cp[m]][stri_temp]['expiry'].values[0] for stri_temp, m, b in
                               zip(stris, mn, map_r)]
                    if min(vols) > 0:
                        res.append([self.spot_price, expirys, stris, vols, mn, mb])
        return res


    def options_model_finder(self):
        model_n = []
        model_b = []
        temp = []
        optionstypes = self.optionsType
        maxquantity = 4
        quantityRange = np.array([i + 1 for i in range(maxquantity)])
        quantityRange = np.append(quantityRange, quantityRange * -1)
        quantity = quantityRange
        for arr in optionstypes:
            k = quantity
            # generate all possible combinations of the integers
            combinations = itertools.product(k, repeat=len(arr))
            for com2 in combinations:
                com1 = com2
                cDivisor = gcd_many(com1)
                tempRe = tuple(np.array(com1) / cDivisor)
                repeat = [arr, tempRe] in temp
                # if repeat:
                #    repeat = model_n[model_b.index(tempRe)] == arr
                if not repeat:  # sum(com1) == sum(np.abs(com1)):
                    # sum1 = sum([abs(comm) for comm in com1])
                    # combo=([np.multiply(com,com1) for com1 in combinations])
                    # for com1 in combo:
                    # result = sum(
                    #    [a * b for a, b in zip(com1, arr)])  # multiply each element by a different integer and sum them up
                    # if result == 0:
                    '''if len(arr[0]) == 2:
                            model_n.append(arr[0])
                            model_b.append(com1)
                        elif len(arr[0]) == 3:
                            if com1[0] == com1[2]:
                                model_n.append(arr[0])
                                model_b.append(com1)
                        else:
                            front = sum([com2 for com2 in com1[:2]])
                            back = sum([com2 for com2 in com1[2:]])
                            front1 = sum([abs(com2) for com2 in com1[:2]])
                            back1 = sum([abs(com2) for com2 in com1[2:]])
                            if front == back and com1[0] * com1[-1] > 0:'''
                    temp.append([arr, com1])
        model_n, model_b = zip(*temp)
        if model_n and model_b:
            unique_k = self.df['K'].unique()



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
                    #loops.append([mn, mb, unique_k.reshape(-1, 1)])
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
            pool = ThreadPoolExecutor(max_workers=8)
            res = []
            for asset in tqdm(pool.map(self.model_find_v2, paras), total=len(paras)):
                res.append(asset)
            df = pd.DataFrame(res, columns=['para','stra', 'loss', 'probal', 'RR', 'wv', 'wp', 'bv', 'bp', 'delta_starting',
                                            'mean_delta', 'std_delta', 'mean_vega', 'mean_theta', 'std_theta',
                                            'premium']).dropna()
            return df
    def model_find_v2(self,paras):
        spot_price, expirys, strs, vols, model, model_b = paras

        preOption, preOption2 = self.preOption,self.preOption2
        quty = [1] * len(strs)
        map_type = {1: 'C', -1: 'P'}
        types = [map_type[i] for i in model]
        quty = [i * j for i, j in zip(quty, model_b)]
        map_fu = [1 if qu > 0 else -1 for qu in model_b]
        map_bs = {1: 'buy_', -1: 'sell_'}
        # Define call and put options
        options = [{'type': type1, 'strike': str1, 'expiry': expiry, 'vol': vol, 'quantity': quty_1, 'map': map_1}
                   for type1, str1, vol, quty_1, map_1, expiry in zip(types, strs, vols, quty, map_fu, expirys)]

        payoffs = calculate_strategy_expiry_payoff(options,preOption2,map_bs)
        net_premium = calculate_strategy_net_premium(options, spot_price).values[0][0]
        premium = calculate_strategy_premium(options, spot_price).values[0][0]
        profit = len(payoffs[payoffs > net_premium])
        loss = len(payoffs[payoffs < net_premium])
        probal = profit / (profit + loss)
        RR = abs(max(payoffs) - net_premium) / abs(net_premium - min(payoffs))
        deltas = calculate_strategy_delta(options,preOption,map_bs)
        vegas = calculate_strategy_vega(options,preOption,map_bs)
        thetas = calculate_strategy_theta(options,preOption,map_bs)
        delta_starting_point = calculate_strategy_delta_starting_point(options,spot_price).values[0][0]
        min_payoffs = min(payoffs)
        min_index = np.where(payoffs == min(payoffs))[0]
        max_index = np.where(payoffs == max(payoffs))[0]
        wv, wp, bv, bp = calculate_strategy_worstBestCase(min_index, max_index, options, preOption)

        straSym = '__'.join([map_type[cp]+'_'+str(int(k))+'_'+str(quty)   for k,cp,quty in zip(strs,model,model_b)])
        return (
            [spot_price, expirys, strs, vols, model, model_b],straSym, abs(min_payoffs - net_premium) / abs(net_premium), probal, RR,
            wv, wp, bv, bp, abs(delta_starting_point),np.mean([abs(i) for i in deltas]), np.std(deltas), np.mean(vegas),
             np.mean(thetas), np.std(thetas), premium)
    def model_plot(self,paras):
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
        options = [{'type': type1, 'strike': str1, 'expiry': expiry, 'vol': vol, 'quantity': quty_1, 'map': map_1}
                   for type1, str1, vol, quty_1, map_1, expiry in zip(types, strs, vols, quty, map_fu, expirys)]

        prices = [calculate_price(option_1, spot_price) for option_1 in options]
        payoffs = calculate_strategy_expiry_payoff(options, preOption2, map_bs)
        net_premium = calculate_strategy_net_premium(options, spot_price).values[0][0]
        premium = calculate_strategy_premium(options, spot_price).values[0][0]
        min_payoffs = min(payoffs)
        deltas = calculate_strategy_delta(options, preOption, map_bs)
        vegas = calculate_strategy_vega(options, preOption, map_bs)
        thetas = calculate_strategy_theta(options, preOption, map_bs)
        min_index = np.where(payoffs == min(payoffs))[0]
        max_index = np.where(payoffs == max(payoffs))[0]
        profit = len(payoffs[payoffs > net_premium])
        loss = len(payoffs[payoffs < net_premium])
        probal = profit / (profit + loss)
        RR = abs(max(payoffs) - net_premium) / abs(net_premium - min(payoffs))
        wv, wp, bv, bp = calculate_strategy_worstBestCase(min_index, max_index, options, preOption)
        # Plot strategy metrics

        per_ivs = preOption2[options[0]['type']][options[0]['strike']]['per_ivs'].astype('float')
        spot_per = preOption2[options[0]['type']][options[0]['strike']]['spot_price'].astype('float').values

        temp_df = pd.DataFrame({'payoffs': payoffs - net_premium, 'spot_per': spot_per, 'per_ivs': per_ivs})
        ax = sns.lineplot(x=spot_per, y=payoffs)
        ax.axhline(net_premium, color='red')

        plt.xlabel('Spot Price')
        plt.ylabel('payoff')
        plt.show()

        print(prices)
        print('net premium ' + str(net_premium))
        print('premium ' + str(premium))
        print('Risk Reward ' + str(RR))
        print('probal ' + str(probal))
        print('lowest possible premium ' + str(min_payoffs))
        print('max loss ' + str(abs(min_payoffs - net_premium) / [abs(net_premium)]))
        print('worst case, vol: ' + str(wv) + ' price: ' + str(wp))
        print('best case, vol: ' + str(bv) + ' price: ' + str(bp))
        print('min thetas ' + str(min(thetas)))
        # print(temp, time_to_expiry, strs, vols, buy, model, model_b)
        print(options)