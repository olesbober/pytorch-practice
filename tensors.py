# practice using some tensors here

import math
import torch
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu") # run with CPU
device = torch.device("cuda:0") # run with GPU

# create input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.cos(x) + torch.sin(x)

# randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

# train by manually getting the gradients
num_epochs, learning_rate = 9001, 10**-6
for ne in range(num_epochs):
    # get prediction
    y_hat = a*x**3+b*x**2+c*x+d

    # compute and print loss
    loss = torch.sum((y-y_hat)**2)
    print("Epoch " + str(ne+1) + ": " + str(loss))

    # compute partials
    par_y_hat = -2.0*(y-y_hat)
    par_a = torch.sum(par_y_hat*x**3)
    par_b = torch.sum(par_y_hat*x**2)
    par_c = torch.sum(par_y_hat*x)
    par_d = torch.sum(par_y_hat)

    # update rules
    a -= learning_rate*par_a
    b -= learning_rate*par_b
    c -= learning_rate*par_c
    d -= learning_rate*par_d

# print final polynomial
print(f'Result: y = {a}x^3 + {b}x^2 + {c}x + {d}')
y_hat = a*x**3+b*x**2+c*x+d

# convert tensor to numpy
x = x.cpu().detach().numpy()
y = y.cpu().detach().numpy()
y_hat = y_hat.cpu().detach().numpy()

# plot it against the actual y
plt.plot(x, y, x, y_hat)
plt.title("Initial Data and Fitted Polynomial: Manual Gradients")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["y", "y_hat"])
plt.show()