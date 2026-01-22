import numpy as np
from statistics import mean, median, mode, StatisticsError

x = np.array([1, 2, 3, 4])
y = np.array([20, 40, 50, 60])

x_mean = mean(x)
y_mean = mean(y)

x_median = median(x)
y_median = median(y)

try:
    x_mode = mode(x)
except StatisticsError:
    x_mode = "No mode"

try:
    y_mode = mode(y)
except StatisticsError:
    y_mode = "No mode"

x_variance = np.var(x)
y_variance = np.var(y)

x_std = np.std(x)
y_std = np.std(y)

covariance = np.cov(x, y, bias=True)[0][1]

print("Mean:", x_mean, y_mean)
print("Median:", x_median, y_median)
print("Mode:", x_mode, y_mode)
print("Variance:", x_variance, y_variance)
print("Standard Deviation:", x_std, y_std)
print("Covariance:", covariance)
