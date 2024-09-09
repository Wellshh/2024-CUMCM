import matplotlib.pyplot as plt
import numpy as np
import random 
from deap import base
from deap import creator
from deap import tools
import math
from pyDOE import lhs
from scipy.stats import norm

# CDF of bad rate

def normal_dist_calculate(lower_bound, upper_bound):
    sigma = 5.10
    mu = 10
    normal_dist = norm(loc=mu, scale=sigma)
    prob = normal_dist.cdf(upper_bound) - normal_dist.cdf(lower_bound)
    return prob

prob = []

for i in range(21):
    prob.append(normal_dist_calculate(i,i+2))

    

def calculate_probability_bad_rate(x):
    return prob[math.floor(x)]


# Basic figures
acc_num = 100 #零件个数d
type_acc = 8 #零件种类
step_num = 2 #工序数
group_num = math.ceil(math.pow(type_acc,1/step_num))
# acc_bad_rate = np.random.randint(5,21,size=type_acc)/100
acc_buy_price = np.random.randint(1,21,size=type_acc)
acc_check_price = np.random.randint(1,11,size=type_acc)
group_num

def calculate_half_num(step_num,group_num):
    result = 0
    for i in range(1, step_num):
        result += group_num**i
    return result

half_num = calculate_half_num(step_num,group_num)
half_check_price = np.random.randint(1,6,size=half_num+1) #半成品检测价， 是否需要增加数组长度方便计算
half_check_price
half_load_price = np.random.randint(7,16,size=half_num+1)
half_seperate_price = np.random.randint(7,10,size=half_num+1)
# half_bad_rate = np.random.randint(5,21,size=half_num+1)/100
product_sale_price = np.random.randint(50,101)
product_check_price = np.random.randint(1,6)
# product_bad_rate = np.random.randint(5,21)/100
product_seperate_price = np.random.randint(7,10)
product_load_price = np.random.randint(7,16)
product_change_price = np.random.randint(1,6)
bad_rate_num = half_num + type_acc + 1

# Fitness Function
creator.create("FitnessMulti",base.Fitness,weights=(0.02,0.8,-0.13))
creator.create("Individual",list,fitness=creator.FitnessMulti)

def calculate_individual_size():
    return type_acc + half_num*2 + 2 #individual构成[零件检修type_acc,半成品检修half_num，半成品拆分half_num，成品检修1,成品拆分1]

IND_SIZE = calculate_individual_size()
toolbox = base.Toolbox()
toolbox.register("attr_binary",random.randint,0,1)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_binary,n=IND_SIZE)

# Population
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

# Evaluation
def evaluate(individual, new_acc_badrate, new_half_badrate, new_product_badrate): #需要修改里面涉及次品率的值
# 零件到半成品
    acc2half_buy_cost = acc_num * sum(acc_buy_price)
    acc2half_check_cost = 0
    for i in range(type_acc):
        acc2half_check_cost += acc_check_price[i] * individual[i]
    acc2half_check_cost *= acc_num
    A = np.zeros((type_acc + 1, type_acc + 1)) #代表每个阶段的不合格品
    B = np.zeros((type_acc + 1, type_acc + 1)) #代表每个阶段的合格品
    M = np.zeros((type_acc + 1, type_acc + 1)) #代表每个阶段拆解可以节省的费用
    half_num_1 = group_num ** (step_num - 1)
    for i in range(1,half_num_1 + 1):
        start_index = i * group_num - group_num
        end_index = i * group_num
        max_result = -math.inf
        min_result = math.inf
        for j in range(start_index,end_index):
            #有可能最后一个半成品合成零件数不够
            if j >= type_acc:
                break
            max_result = max(max_result,new_acc_badrate[j] * (1 - individual[j]))
            min_result = min(min_result,1 - new_acc_badrate[j] * individual[j])
        A[1][i] = max_result * acc_num
        B[1][i] = min_result * acc_num
    acc2half_load_cost = 0
    acc2half_seperate_cost = 0
    half1_check_cost = 0
    for i in range(1, half_num_1 + 1):
        acc2half_load_cost += half_load_price[i] * (A[1][i] + B[1][i])
        acc2half_seperate_cost += individual[type_acc + i - 1] * half_seperate_price[i] * individual[type_acc + half_num + i - 1] * (A[1][i] + B[1][i] * new_half_badrate[i])
        half1_check_cost += half_check_price[i] * individual[type_acc + half_num + i - 1] * (A[1][i] + B[1][i])
    # 零件到半成品，拆解能省下的费用
    acc2half_save_cost = 0
    for i in range(1, half_num_1 + 1):
        start_index = i * group_num - group_num
        end_index = i * group_num
        acc_sum = 0
        for j in range(start_index,end_index):
            #有可能最后一个半成品合成零件数不够
            if j >= type_acc:
                break
            acc_sum += acc_buy_price[j]
            bad_ones = min(B[1][i],acc_num * new_acc_badrate[j] * (1 - individual[j])) * acc_buy_price[j]
        M[1][i] = (B[1][i] * new_half_badrate[i] * acc_sum + bad_ones ) * individual[type_acc + i - 1]
        acc2half_save_cost += M[1][i] * individual[type_acc + half_num + i - 1]
# 半成品到半成品,递推关系

    def calculate_past_half_num(i,group_num, step_num):
        result = 0
        for j in range(1,i):
            result += group_num ** (step_num - j)
        return result

    
    half_check_sum = 0
    half_separate_sum = 0
    half_load_sum = 0
    half_save_cost = 0

    for i in range(2,step_num):
        half_num_i = group_num ** (step_num - i)
        past_half_num = calculate_past_half_num(i - 1,group_num,step_num) #上一步已经计算的半成品数
        temp_check_sum = 0
        temp_load_sum = 0
        temp_separate_sum = 0
        temp_save_cost = 0
        for j in range(1,half_num_i + 1):
            start_index = j * group_num - group_num + 1
            end_index = j * group_num
            max_result = -math.inf
            min_result = math.inf
            M_sum = 0
            for k in range(start_index,end_index):
                if k >= half_num_i * group_num:
                    break
                max_result = max(max_result, (A[i-1][k] + B[i-1][k] * new_half_badrate[past_half_num + k] ) * (1 - individual[type_acc + past_half_num + k - 1]))
                min_result = min(min_result, B[i-1][k] * (1 - new_half_badrate[past_half_num + k]))
                M_sum += M[i-1][k]
            A[i][j] = max_result
            B[i][j] = min_result
            temp_check_sum += (A[i][j] + B[i][j]) * half_check_price[past_half_num + half_num_i * group_num + j] * individual[type_acc + past_half_num + half_num_i * group_num + j - 1]
            temp_load_sum += (A[i][j] + B[i][j]) * half_load_price[past_half_num + half_num_i * group_num + j]
            temp_separate_sum += individual[type_acc + past_half_num + half_num_i * group_num + j - 1] * half_seperate_price[past_half_num + half_num_i * group_num + j] * individual[type_acc + past_half_num + half_num_i * group_num + half_num + j - 1] * (A[i][j] + B[i][j] * new_half_badrate[past_half_num + half_num_i * group_num + j])
            start_index = j * group_num ** i - group_num ** i
            end_index = j * group_num ** i
            tt_sum = 0
            for k in range(start_index,end_index):
                if k >= type_acc:
                    break
                tt_sum += acc_buy_price[k]
            M[i][j] = (M_sum + tt_sum * B[i][j] * new_half_badrate[past_half_num + half_num_i * group_num + j] * individual[type_acc + past_half_num + half_num_i * group_num + j - 1])
            temp_save_cost += M[i][j] * individual[type_acc + past_half_num + half_num_i * group_num + half_num + j - 1]
        half_check_sum += temp_check_sum
        half_load_sum += temp_load_sum
        half_separate_sum += temp_separate_sum
        half_save_cost += temp_save_cost
    
# 半成品到成品
    past_half_num = calculate_past_half_num(step_num - 1,group_num,step_num)
    max_result = -math.inf
    min_result = math.inf
    M_sum = 0
    for i in range(1, group_num + 1):
        max_result = max(max_result, (A[step_num - 1][i] + B[step_num - 1][i] * new_half_badrate[past_half_num + i]) * (1 - individual[type_acc + past_half_num + i - 1]))
        min_result = min(min_result, B[step_num - 1][i] * (1 - new_half_badrate[past_half_num + i]))
        M_sum += M[step_num - 1][i]
    A[step_num][1] = max_result
    B[step_num][1] = min_result
    product_check_cost = (A[step_num][1] + B[step_num][1]) * product_check_price * individual[-2]
    product_load_cost = (A[step_num][1] + B[step_num][1]) * product_load_price
    product_separate_cost = individual[-2] * product_seperate_price * individual[-1] * (A[step_num][1] + B[step_num][1] * new_product_badrate)
    product_save_cost = (M_sum + acc2half_buy_cost * B[step_num][1] * new_product_badrate * individual[type_acc + past_half_num] ) * individual[-1]
    product_change_cost = product_change_price * individual[-2] * (A[step_num][1] + B[step_num][1] * new_product_badrate)
    product_sale = product_sale_price * (A[step_num][1] + B[step_num][1]) * (1 - new_product_badrate * individual[-1])
    

# 总费用
    total_cost = acc2half_buy_cost + acc2half_check_cost + acc2half_load_cost + acc2half_seperate_cost + half1_check_cost - acc2half_save_cost + half_check_sum + half_load_sum + half_separate_sum - half_save_cost + product_check_cost + product_load_cost + product_separate_cost - product_save_cost + product_change_cost
    profit = product_sale - total_cost

# 总检验次数
    total_check = acc_num * sum(individual[:type_acc]) + sum(individual[-2:])
    for i in range(1,step_num):
        half_num_i = group_num ** (step_num - i)
        for j in range(1,half_num_i + 1):
            step = calculate_past_half_num(i,group_num,step_num)
            total_check += (A[i][j] + B[i][j]) * individual[type_acc + step + j - 1] + (A[i][j] + B[i][j] * new_half_badrate[calculate_past_half_num(i,group_num,step_num) + j]) * individual[type_acc + step + half_num + j - 1]

# 成品合格总数
    total_product = B[step_num][1] * (1 - new_product_badrate) 
    return total_product, profit, total_check



num_population = 100
N_GEN = 1000
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

pop = toolbox.population(n=num_population)

#适应度计算
sample_num = 2

def generate_new_bad_rate(ind):
    #拉丁超立方采样得到新的次品率
    samples = lhs(bad_rate_num, samples=sample_num)
    param_ranges = [0,20]
    scaled_samples = samples.T * (param_ranges[1] - param_ranges[0]) + param_ranges[0]
    sample_prob_list = []
    fit_list = []
    for i in range(sample_num):
        new_acc_badrate = scaled_samples[:type_acc,i] / 100
        new_half_badrate = scaled_samples[type_acc:type_acc + half_num,i] / 100
        new_product_badrate = scaled_samples[-1,i] / 100
        temp_multi = 1
        for j in range(bad_rate_num):
            temp_multi *= calculate_probability_bad_rate(scaled_samples[j][i])
        # print(temp_multi)
        sample_prob_list.append(temp_multi)
        new_half_badrate_list = new_half_badrate.tolist()
        new_half_badrate_list.insert(0,0)
        temp_fit = toolbox.evaluate(ind, new_acc_badrate, new_half_badrate_list, new_product_badrate)
        fit_list.append(temp_fit)
    # 重新计算概率
    sum_pro = sum(sample_prob_list)
    fit = [0,0,0]
    for i in range(sample_num):
        sample_prob_list[i] = sample_prob_list[i] / sum_pro
        for j in range(3):
            fit[j] += sample_prob_list[i] * fit_list[i][j]
    # print(sample_prob_list)
    return fit[0],fit[1],fit[2]

CXPB, MUTPB = 0.5, 0.2

# 生成父代个体
for ind in pop:
    ind.fitness.values = generate_new_bad_rate(ind)


for g in range(N_GEN):
    # 选择并克隆个体
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    if invalid_ind:
        fitnesses = map(generate_new_bad_rate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

    # 将父代与子代合并，选择下一代
    pop = toolbox.select(offspring + pop, num_population)

    fits = [ind.fitness.values[0] for ind in pop]
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    print("Generation %i: Min %s Max %s Avg %s Std %s" % (g, min(fits), max(fits), mean, std))
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    print("Pareto Front (Generation %i):" % g)
    for ind in pareto_front:
        print(ind, ind.fitness.values)
    best_individual = max(pop, key=lambda ind: ind.fitness.values)
    print("Best individual (Generation %i): %s, Fitness: %s" % (g, best_individual, best_individual.fitness.values))
    



