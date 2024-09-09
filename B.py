# Simulated Anneling Algorithm 解决问题二
import random
import math
num_accessory = 100
bad_rate = [0.1,0.1,0.1]
cost_list_accessory = [2,3]
product_list = [6,3]
failed_product_list = [6,5]
sale = 56
buy_list = [4,18]



def profit(X):
    buy_accessory = num_accessory * (cost_list_accessory[0] + cost_list_accessory[1])
    check_1 = X[0] * num_accessory * bad_rate[0] + X[1] * num_accessory * bad_rate[1]
    left_1 = max(1-X[0]*bad_rate[0],1-X[1]*bad_rate[1]) * num_accessory #一阶段剩下的
    check_2 = product_list[0] * left_1# 装配费用
    check_3 = X[2] * left_1 * product_list[1] #检验成品
    no_product = left_1 - (1 - bad_rate[2]) * (left_1 -((1-X[0]) * num_accessory * bad_rate[0] + (1-X[1]) * num_accessory * bad_rate[1]))
    part = X[2] * X[3] * no_product * failed_product_list[1] #拆解费用
    exchange = (1 - X[2]) * (left_1 -((1-X[0]) * num_accessory * bad_rate[0] + (1-X[1]) * num_accessory * bad_rate[1]) ) * (failed_product_list[0] + buy_list[0] + buy_list[1] + product_list[0] )
    n_3 =  left_1 * (1 - bad_rate[2] * X[3]) -  (1- bad_rate[2]) * X[3] * ((1-X[0]) * num_accessory * bad_rate[0] + (1-X[1]) * num_accessory * bad_rate[1]) #流入市场的产品数量
    income = sale * n_3
    part_income = (cost_list_accessory[0] + cost_list_accessory[1]) * (1 - bad_rate[2]) * X[2] * X[3] * (left_1 -((1-X[0]) * num_accessory * bad_rate[0] + (1-X[1]) * num_accessory * bad_rate[1])) #合格拆解利润
    part_income_ = (cost_list_accessory[0] * (1-X[0]) * num_accessory * bad_rate[0] + cost_list_accessory[1] * (1-X[1]) * num_accessory * bad_rate[1])
    return income - (buy_accessory + check_1 + check_2 + check_3 + part + exchange) + part_income + part_income_







def SA():
    T = 1000
    T_min = 1
    alpha = 0.9
    X = [0,0,0,0]
    for j in range(4):
        X[j] = random.randint(0,1)
    while T > T_min:
        for i in range(100):
            E = profit(X)
            X_new = X.copy()
            X_new[random.randint(0,3)] = 1 - X_new[random.randint(0,3)]
            E_new = profit(X_new)
            if E_new > E:
                X = X_new
            else:
                if E - E_new <= 500:
                    if random.random() < math.exp((E - E_new) / T):
                        X = X_new
            print(X)
        print(T)
        print(profit([0,0,1,0]))
        T = T * alpha
    return X

print(SA())






