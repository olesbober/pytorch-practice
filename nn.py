# nn.py
# use the nn package in PyTorch to build a neural network

import math
import matplotlib.pyplot as plt
import torch

# create input and output data
x = torch.linspace(-math.pi, math.pi, 2000)
y = 3*x**5+2*x**3+torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p) # unsqueeze(-1) is the transpose of x, and .pow(p) makes each column the power of the elements in p

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

# train by using nn package
num_epochs, learning_rate = 9001, 10**-6
for ne in range(num_epochs):
    # Forward pass: compute predicted y by passing xx to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_hat = model(xx)

    # compute and print loss
    loss = loss_fn(y_hat, y)
    print("Epoch " + str(ne+1) + ": " + str(loss))

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
y_hat = linear_layer.weight[:, 2].item()*x**3+linear_layer.weight[:, 1].item()*x**2+linear_layer.weight[:, 0].item()*x+linear_layer.bias.item()

# plot it against the actual y
plt.plot(x, y, x, y_hat)
plt.title("Initial Data and Fitted Polynomial: Defined Auto Gradients")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["y", "y_hat"])
plt.show()