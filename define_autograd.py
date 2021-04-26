# define_autograd.py
# in this example, we will be writing our own autograd functions

import math
import matplotlib.pyplot as plt
import torch

class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        ctx.save_for_backward(input)
        return 0.5*(5*input**3-3*input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        input, = ctx.saved_tensors
        return grad_output*1.5*(5*input**2-1)

dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0")

# create input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.cos(x)**2

# randomly initialize weights
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

# train by using autograd
num_epochs, learning_rate = 9001, 10**-6
for ne in range(num_epochs):
    # to apply our function, we use the .apply method
    P3 = LegendrePolynomial3.apply

    # forward pass: compute predicted y using operations; we compute
    # P3 using our custom autograd operation.
    y_hat = a + b * P3(c + d * x)

    # compute and print loss
    loss = torch.sum((y-y_hat)**2)
    print("Epoch " + str(ne+1) + ": " + str(loss))

    # compute backward pass
    loss.backward()

    # update weights using gradient descent
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
plt.title("Initial Data and Fitted Polynomial: Defined Auto Gradients")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["y", "y_hat"])
plt.show()