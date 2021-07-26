### Line of best fit y = mx + b
### two dimessional Linear Regression

# x = x intercept
#m = slope of line
#b = Y intercept

#M = (Mean of X) * (Mean of Y) - Mean X*Y divide by mean x squared - mean x squared
# b = (mean of y) - m * (mean of x)
# squared error - rsquared
# rsquared = 1 - SE regression line/ SE Mean(ys)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_datset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys))/
         ((mean(xs)* mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_datset(40,40,2, correlation='pos')

m,b = best_fit_slope_and_intercept(xs,ys)
print(m,b)

regression_line = [(m*x) + b for x in xs]
predict_x = 8
predict_y = (m*predict_x) + b
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)


plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs,regression_line)
plt.show()







