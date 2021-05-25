import torch


def _softplus(x):
    return torch.log1p(torch.exp(x))


def shifted_softplus(x):
    """
    Softplus nonlinearity shifted by -log(2) such that shifted_softplus(0.) = 0.

    y = log(0.5e^x + 0.5)

    """
    y = torch.where(x < 14., _softplus(torch.where(x < 14., x, torch.zeros_like(x))), x)
    return y - torch.log(torch.tensor(2.))
