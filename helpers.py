"""THWS/MAI/ANN - Assignment 1 - helper functions

Created: Magda Gregorova, 9/5/2024
"""


import csv
import torch
from ann_code.linear_regression import mse_forward
from ann_code.layers import Linear, Relu


def load_data(filename='./ann_data/toy_data.csv'):
    with open(filename, newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        dset = []
        for row in reader:
            dset.append(row)
        dtensor = torch.tensor(dset)
        means, stds = dtensor.mean(0), dtensor.std(0)
        dtensor = (dtensor - means) / stds
    
    return (dtensor[:, :3], dtensor[:, -2:-1])


def numerical_gradient(f, x, h=1e-4):
    xflat = x.flatten()
    xg = torch.zeros_like(xflat)

    num_dim = xg.size(0)
    for i in range(num_dim):
        xu, xl = xflat.clone(), xflat.clone()
        xu[i] += h
        xl[i] -= h
        xg[i] = (f(xu.view(*x.shape)) - f(xl.view(*x.shape)))/(2*h)

    return xg.view(*x.shape)


def grad_checker(grad_analytic, grad_numeric, rnd=False):
    if rnd:
        # print out random elements of compared vectors
        idx = torch.randint(high=grad_analytic.numel(), size=(1, 5))
        grad_analytic = torch.take(grad_analytic, idx)
        grad_numeric = torch.take(grad_numeric, idx)
        print("To save space, printing only randomly selected elements:")
        print(f"analytic grad: {grad_analytic}")
        print(f"numerical grad: {grad_numeric}")
        print(f"relative error: {abs(grad_analytic - grad_numeric) / 0.5*abs(grad_analytic - grad_numeric)}")
    else:
        # print out all elements of compared gradients
        print(f"analytic grad: {grad_analytic}")
        print(f"numerical grad: {grad_numeric}")
        print(f"relative error: {abs(grad_analytic - grad_numeric) / 0.5*abs(grad_analytic - grad_numeric)}")


def grad_model(model, inputs, labels):
    """Check gradient of mse_loss with respect to first layer parameters."""

    # mdl = copy.deepcopy(model)
    mdl = model

    # autograd record operations
    for layer in mdl.layers:
        if isinstance(layer, Linear):
            layer.W.requires_grad_()
            layer.b.requires_grad_()

    inputs.grad = torch.zeros_like(inputs)
    inputs.requires_grad_()
    preds = mdl.forward(inputs)
    loss, _ = mse_forward(preds, labels)
    loss.backward()
    return mdl 	


def check_architecture(model):
    """Quick and dirty checker for the model architecture."""

    wrong_architecture = False
    n_inst, n_dim = model.layers[0].ins.shape

    # check num layers
    if len(model.layers) != 7:
        wrong_architecture = True
    if not isinstance(model.layers[0], Linear):
        wrong_architecture = True
    if not isinstance(model.layers[2], Linear):
            wrong_architecture = True
    if not isinstance(model.layers[4], Linear):
            wrong_architecture = True
    if not isinstance(model.layers[4], Linear):
            wrong_architecture = True
    if not isinstance(model.layers[1], Relu):
        wrong_architecture = True
    if not isinstance(model.layers[3], Relu):
            wrong_architecture = True
    if not isinstance(model.layers[5], Relu):
            wrong_architecture = True
    if model.layers[1].ins.shape != torch.zeros(n_inst, 5).shape:
            wrong_architecture = True
    if model.layers[3].ins.shape != torch.zeros(n_inst, 10).shape:
            wrong_architecture = True
    if model.layers[5].ins.shape != torch.zeros(n_inst, 4).shape:
            wrong_architecture = True

    if wrong_architecture:
        print(f'Your NN architecture definitions seems WRONG!')
    else:
        print(f'Your NN architecture definitions seems CORRECT.')




