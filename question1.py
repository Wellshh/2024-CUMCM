import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp	
import pandas as pd

form2_path = "form2.xlsx"
form1_path = "form1.xlsx"
form2 = pd.read_excel(form2_path, sheet_name='2023年统计的相关数据')
# print(form2.head())
form1 = pd.read_excel(form1_path, sheet_name='乡村种植的农作物')
# 导入农作物信息
crop_num_name = {}
crop_name_num = {}
for index, row in form1.iterrows():
    if index <= 40:
        crop_id = row['作物编号']
        crop_name = row['作物名称']
        crop_num_name[crop_id] = crop_name
        crop_name_num[crop_name] = crop_id
    else: break
# print(crop_num_name)
# print(crop_name_num)

# 导入地块类型
form4 = pd.read_excel(form1_path, sheet_name='乡村的现有耕地')
field_num_type = {}
for index, row in form4.iterrows():
    field_num_type[row['地块名称']] = (row['地块类型'],row['地块面积/亩']) 
print(field_num_type)
y = {} #亩产量
c = {} #种植成本
p = {} #销售单价

for index, row in form2.iterrows():
    if index <= 106:
        crop_id = crop_name_num[row['作物名称']]
        land_type = row['地块类型']
        season = row['种植季次']
    
    # 键使用 (crop_id, land_type, season) 的组合来唯一确定变量
        key = (crop_id, land_type, season)
    
    # 亩产量转换为吨 (1 吨 = 2000 斤)
        y[key] = row['亩产量/斤']
    
    # 种植成本 (元/亩)
        c[key] = row['种植成本/(元/亩)']
    
    # 销售单价范围，取均值
        price_range = row['销售单价/(元/斤)']
        low_price, high_price = map(float, price_range.split('-'))
        p[key] = (low_price + high_price) / 2.0
        if land_type == '智慧大棚' and season == '第二季':
            y[(crop_id,land_type,'第一季')] = row['亩产量/斤']
        else:
            y[(crop_id,land_type,'第二季')] = row['亩产量/斤']
    else: break

# 2023年种植情况
form3 = pd.read_excel(form2_path, sheet_name='2023年的农作物种植情况')
print(form3.head())
# 计算预期销售量
expected_sales = [0 for _ in range(len(crop_name_num))]
for index, row in form3.iterrows():
    expected_sales[row['作物编号'] - 1] += row['种植面积/亩'] * y[(row['作物编号'], field_num_type[row['种植地块']][0], row['种植季次'])]
print(expected_sales)

# print(c)
# print(p)
# 创建Gurobi模型
env = gp.Env()
env.setParam('DualReductions', 0)
model_question1_1 = gp.Model("crop_optimization_11",env=env)
# 定义变量
indices = [(crop, field, season)  
        for crop in crop_num_name.keys()  
        for field in field_num_type  
        for season in range(1, 15)]  
x = model_question1_1.addVars(indices, lb=0, vtype=GRB.CONTINUOUS, name="x")
profit = 0
print(y)
for item in indices:
    crop = item[0]
    field = item[1]
    season = item[2]
    if (crop,field_num_type[field][0],season) in y.keys() and (crop,field_num_type[field][0],season) in p.keys() and (crop,field_num_type[field][0],season) in c.keys():
        profit += x[item] * p[(crop,field_num_type[field][0],season)] - c[(crop,field_num_type[field][0],season)] * x[item]
    else:
        profit += 0
# for crop in crop_num_name.keys():
#     crop_production = 0
#     for field in field_num_type:
#         for season in range(1, 15):
#             key = (crop, field, season)
#             if season % 2 == 0:
#                 current_season = '第二季'
#             else: current_season = '第一季'
#             if (crop,field_num_type[field][0],current_season) in y.keys(): 
#                 crop_production += x[key] * y[(crop,field_num_type[field][0], current_season)]
#             else: crop_production += 0
#     if crop_production > expected_sales[crop-1]:
#         final_crop_production = expected_sales[crop-1]
#     else:
#         final_crop_production = crop_production
#     profit += final_crop_production * p[(crop,field_num_type[field][0], current_season)] - c[(crop,field_num_type[field][0], current_season)] * crop_production
# 先初始化 profit
# profit = gp.LinExpr()

# # 添加一个新的变量 z，表示最终的实际产量
# z = model_question1_1.addVars(indices, lb=0, vtype=GRB.CONTINUOUS, name="z")

# # 遍历所有作物
# for crop in crop_num_name.keys():
#     crop_production = gp.LinExpr()  # 用线性表达式来存储 crop_production
#     for field in field_num_type:
#         for season in range(1, 15):
#             key = (crop, field, season)
            
#             # 判断当前是第几季
#             if season % 2 == 0:
#                 current_season = '第二季'
#             else:
#                 current_season = '第一季'
            
#             # 如果在y.keys()中，计算产量
#             if (crop, field_num_type[field][0], current_season) in y.keys():
#                 crop_production += x[key] * y[(crop, field_num_type[field][0], current_season)]
    
#     # 添加约束，确保实际产量 z 小于等于 crop_production 和 expected_sales[crop-1]
#     for field in field_num_type:
#         for season in range(1, 15):
#             if (crop, field_num_type[field][0], current_season) in y.keys():
#                 model_question1_1.addConstr(z[(crop, field, season)] <= crop_production, name=f"z_limit_production_{crop}_{field}_{season}")
#                 model_question1_1.addConstr(z[(crop, field, season)] <= expected_sales[crop-1], name=f"z_limit_sales_{crop}_{field}_{season}")
    
#     # 仅在p和c都有对应作物时计算
#     if (crop, field_num_type[field][0], current_season) in p.keys() and (crop, field_num_type[field][0], current_season) in c.keys():
#         # 使用 LinExpr 包装价格和成本，将其转化为 Gurobi 的线性表达式
#         price = gp.LinExpr(p[(crop, field_num_type[field][0], current_season)])  # 销售价格
#         cost = gp.LinExpr(c[(crop, field_num_type[field][0], current_season)])  # 种植成本

#         # 计算收益，使用 z 变量作为实际产量
#         profit += z[(crop, field, season)] * price - crop_production * cost

# 设置目标函数
model_question1_1.setObjective(profit, GRB.MAXIMIZE)

# 添加约束条件
# 不能超过预期产量


#每块地种植农作物不超过面积
for season in range(1,15):
    for field in field_num_type:
        model_question1_1.addConstr(sum(x[(crop, field, season)] for crop in crop_num_name.keys()) <= field_num_type[field][1])

#农作物种植类型约束
food_crop = []
vegetable_crop = []
fungi_crop = []
bean_crop = [1,2,3,4,5,17,18,19]
for i in range(1,42):
    if i <= 15:
        food_crop.append(i)
    elif i <=37 and i > 16:
        vegetable_crop.append(i)
    elif i >= 38 and i <= 41:
        fungi_crop.append(i)
#平旱地、梯田和山坡地适宜每年种植一季粮食类作物
#水浇地适宜每年种植一季水稻或两季蔬菜，大白菜、白萝卜和红萝卜只能在第二季种植
#普通大棚适宜每年种植一季蔬菜和一季食用菌。普通大棚每年种植两季作物，第一季可种植多种蔬菜（大白菜、白萝卜和红萝卜除外），第二季只能种植食用菌。
#智慧大棚适宜每年种植两季蔬菜，大白菜、白萝卜和红萝卜除外。

for item in indices:
    crop = item[0]
    field = item[1]
    season = item[2]
    if field == '平旱地' or field == '梯田' or field == '山坡地':
        if crop in [vegetable_crop,fungi_crop,16]:
            model_question1_1.addConstr(x[item] == 0)
    elif field == '水浇地':
        if crop in [food_crop,fungi_crop]:
            model_question1_1.addConstr(x[item] == 0)
    elif field == '普通大棚':
        if crop in [food_crop,16]:
            model_question1_1.addConstr(x[item] == 0)
    else:
        if crop in [food_crop,fungi_crop,35,36,37]:
            model_question1_1.addConstr(x[item] == 0)
    
for crop in food_crop:
    for season in range(1,15,2):
        model_question1_1.addConstrs(x[crop,field,season] * x[crop,field,season + 1] == 0 for field in field_num_type if field_num_type[field][0] in ['平旱地', '梯田', '山坡地'])
for season in range(1,15,2):
    model_question1_1.addConstrs(x[16,field, season] * x[16,field, season + 1] == 0 for field in field_num_type if field_num_type[field][0] == '水浇地')

for crop in [fungi_crop,35,36,37]:
    for season in range(1,15,2):
        model_question1_1.addConstrs(x[(crop, field, season)] == 0 for field in field_num_type if field_num_type[field][0] == '普通大棚')
for crop in vegetable_crop:
    for season in range(2,15,2):
        model_question1_1.addConstrs(x[(crop, field, season)] == 0 for field in field_num_type if field_num_type[field][0] == '普通大棚')

for crop in [35,36,37]:
    for season in range(1,15,2):
        model_question1_1.addConstrs(x[(crop, field, season)] == 0 for field in field_num_type if field_num_type[field][0] == '水浇地')



# 白菜和萝卜在第二季每块地只能种植一种
for season in range(2,15,2):
    for crop in vegetable_crop:
        if crop not in [35,36,37]:
            model_question1_1.addConstrs(x[(crop, field, season)] == 0 for field in field_num_type if field_num_type[field][0] == '水浇地')
for season in range(2,15,2):
    for field in field_num_type:
        if field_num_type[field][0] == '水浇地':
            model_question1_1.addConstr(x[(35, field, season)] * x[(36, field, season)] == 0)   
            model_question1_1.addConstr(x[(36, field, season)] * x[(37, field, season)] == 0)


# 任意一块地都不能连续种植
for field in field_num_type:
    for crop in crop_num_name.keys():
        for season in range(1,14):
            model_question1_1.addConstr(x[(crop, field, season)] * x[(crop, field, season + 1)] == 0)

# 因含有豆类作物根菌的土壤有利于其他作物生长，从 2023 年开始要求每个地块（含大棚）的所有土地三年内至少种植一次豆类作物
for crop in bean_crop:
    for season in range(1,9,2):
        model_question1_1.addConstrs(x[(crop, field, season)] + x[(crop, field, season+1)] + x[(crop, field, season+2)] + x[(crop, field, season+3)] + x[(crop, field, season+4)] + x[(crop, field, season+5)] >= field_num_type[field][1] for field in field_num_type)
model_question1_1.optimize()






            



print(x)
    





                





