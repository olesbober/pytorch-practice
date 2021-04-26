# autograd.py

import math
import torch
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0")

# create input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = 1/2*torch.cos(x) + 5*x

# randomly initialize weights
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

# train by using autograd
num_epochs, learning_rate = 9001, 10**-6
for ne in range(num_epochs):
    # get prediction
    y_hat = a*x**3+b*x**2+c*x+d

    # compute and print loss
    loss = torch.sum((y-y_hat)**2)
    print("Epoch " + str(ne+1) + ": " + str(loss))

    # use autograd to compute backward pass
    loss.backward()

    # update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate*a.grad
        b -= learning_rate*b.grad
        c -= learning_rate*c.grad
        d -= learning_rate*d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

# print final polynomial
print(f'Result: y = {a}x^3 + {b}x^2 + {c}x + {d}')
y_hat = a*x**3+b*x**2+c*x+d

# convert tensor to numpy
x = x.cpu().detach().numpy()
y = y.cpu().detach().numpy()
y_hat = y_hat.cpu().detach().numpy()

# plot it against the actual y
plt.plot(x, y, x, y_hat)
plt.title("Initial Data and Fitted Polynomial: Auto Gradients")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["y", "y_hat"])
plt.show()