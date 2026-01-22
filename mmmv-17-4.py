import numpy as np
from statistics import mean, median, mode, StatisticsError

data = np.array([10, 12, 15, 18, 20, 22, 25])

data_mean = mean(data)
data_median = median(data)

try:
    data_mode = mode(data)
except StatisticsError:
    data_mode = "No mode"

data_variance = np.var(data)

print("Mean:", data_mean)
print("Median:", data_median)
print("Mode:", data_mode)
print("Variance:", data_variance)
