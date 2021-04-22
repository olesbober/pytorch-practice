# warmup with numpy

import numpy as np
import math
import matplotlib.pyplot as plt

# create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.cos(x)

# randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

num_epochs, learning_rate = 9001, 10**-6
for ne in range(num_epochs):
    # get y prediction
    y_hat = a*x**3+b*x**2+c*x+d

    # compute and print loss
    loss = np.sum((y-y_hat)**2)
    print("Epoch " + str(ne+1) + ": " + str(loss))

    # get partials with respect to each parameter
    par_y = -2.0*(y-y_hat)
    par_a = np.sum(par_y*x**3)
    par_b = np.sum(par_y*x**2)
    par_c = np.sum(par_y*x)
    par_d = np.sum(par_y)

    # update rules
    a = a-learning_rate*par_a
    b = b-learning_rate*par_b
    c = c-learning_rate*par_c
    d = d-learning_rate*par_d

# print final polynomial
print(f'Result: y = {a}x^3 + {b}x^2 + {c}x + {d}')
y_hat = a*x**3+b*x**2+c*x+d

# plot it against the actual y
plt.plot(x, y, x, y_hat)
plt.title("Initial Data and Fitted Polynomial")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["y", "y_hat"])
plt.show()