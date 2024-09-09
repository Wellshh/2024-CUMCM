import math
from scipy.stats import norm

def normal_dist_calculate(lower_bound, upper_bound):
    sigma = 5.10
    mu = 10
    normal_dist = norm(loc=mu, scale=sigma)
    prob = normal_dist.cdf(upper_bound) - normal_dist.cdf(lower_bound)
    return prob

prob = []

for i in range(21):
    prob.append(normal_dist_calculate(i,i+2))


print(normal_dist_calculate(9,11))
print(prob)
