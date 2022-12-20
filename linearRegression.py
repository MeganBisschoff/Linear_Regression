# ----- INSTRUCTIONS ----- #
'''
In this task you will use the _db dataset mentioned above to perform
  linear regression to find the best fit line through the data.
Reserve the last 20 observations for testing and use the rest for training
  your model.
Instead of using linear_model.LinearRegression() from sklearn, write a
  function and make use of numpy to calculate the gradient and the
  y-intercept of the best fit line, which has equation 𝑦 = 𝑚𝑥 + 𝑏. 
  The equations below describe how both the gradient and the y-intercept can
  be calculated from the training data and labels. 
  Note: when you calculate the gradient, you will need to reshape the 
  x array to remove an extra dimension of 1 from its shape (it has this 
  as the dataset was formatted for use with the sklearn functions, which 
  require this extra dimension). 
  You can easily do this by applying .squeeze() to the x array when you pass 
  it as an argument to the method. Hint: if the line doesn’t look like it 
  fits the data well, there is a bug in your code.
      𝑚 = (μ(𝑥) * μ(𝑦) − μ(𝑥 * 𝑦))/((μ(𝑥))2 − μ(𝑥2))
      𝑏 = μ(𝑦) − 𝑚 * μ(𝑥)
      𝑊ℎ𝑒𝑟𝑒 μ 𝑖𝑠 𝑎 𝑚𝑒𝑎𝑛 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛
Use these values to produce a figure with the following:
  Scatter plot of training data colored red.
  Scatter plot of testing data colored green.
  Line graph for the best-fit line colored blue.
  Legend. 
'''

# ----- TASK ----- #

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

# Initialise the dataset.
diabetes = load_diabetes()

# Initialise label of the dataset, ie. the bmi. 
bmi_value = diabetes.data[:, 2] 

# Split the data into training and testing (last 20) sets.
x_train = bmi_value[:-20]
x_test = bmi_value[-20:]

# Split the targets into training and testing (last 20) sets
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

# Regression function to return the slope (m) and intercept (b) of y.
def line_of_best_fit(x_array, y_array):

    # Formula 𝑚 = (μ(𝑥) * μ(𝑦) − μ(𝑥 * 𝑦)) / ((μ(𝑥))2 − μ(𝑥2))
    m = ((np.mean(x_array) * np.mean(y_array)) - np.mean(x_array * y_array)) /  (np.mean(x_array)**2 - np.mean(x_array**2))
    # Formula 𝑏 = μ(𝑦) − 𝑚 * μ(𝑥)
    b = np.mean(y_array) - m * np.mean(x_array)
    return m, b

# Call function to train data.
m_value, b_value = line_of_best_fit(x_train, y_train)

# Determine line of best fit with formula 𝑦 = 𝑚𝑥 + 𝑏
line = (m_value * x_test) + b_value

# Plot the training data, in red.
plt.scatter(x_train, y_train, c='r')

# Plot the testing data, in green.
plt.scatter(x_test, y_test, c='g')

# Plot the line of best fit, in blue.
plt.plot(x_test, line, c='b')

# Add annotaions and show graph
plt.title("Diabetes Progression")
plt.legend(["Training Data", "Testing Data", "Line Of Best Fit"])
plt.show()


# ----- RESOURCES ----- #
'''
Diabetes dataset variable (BMI) learnt at
    https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
Assign colours to scatter function learnt at
    https://www.mathworks.com/help/matlab/ref/scatter.html
Adding legend learnt at
    https://www.geeksforgeeks.org/matplotlib-pyplot-legend-in-python/
'''