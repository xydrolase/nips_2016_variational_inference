#!/usr/bin/env python

import numpy as np
import scipy.stats
import torch
from matplotlib import pyplot as plt


def contour_plot_mvn(mu, sigma, n_grid=100, **kwargs):
    # some heuristics: generate a meshgrid with a bounding box computed by
    # mu +- 5 * sigma on each coordinate.
    assert mu.shape[0] == sigma.shape[0] == 2

    rv = scipy.stats.multivariate_normal(mu.flatten(), sigma)

    xmin, ymin = mu - np.sqrt(np.diag(sigma)) * 5
    xmax, ymax = mu + np.sqrt(np.diag(sigma)) * 5

    x, y = np.meshgrid(
        np.linspace(xmin, xmax, n_grid),
        np.linspace(ymin, ymax, n_grid)
    )

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    plt.contour(x, y, rv.pdf(pos), **kwargs)


def mvn_posterior(mu0, sigma0, sigma, X):
    """
    Compute the posterior distribution p(z|x), for a MVN prior p(z) with parameters
    mu0 and sigma0, and the likelihood p(x|z) with mean z, and a known covariance matrix sigma,
    using the conjugacy.

    The observed data is given by the matrix X.

    :param mu0:
    :param sigma0:
    :param sigma:
    :param X:
    :return: The mean vector and the covariance matrix of p(z|x)
    """
    inv_sigma0 = np.linalg.inv(sigma0)
    inv_sigma = np.linalg.inv(sigma)
    cov_pos = np.linalg.inv(
        inv_sigma0 + X.shape[0] * inv_sigma
    )

    if isinstance(X, torch.Tensor):
        X = X.numpy()

    x_mean = np.mean(X, axis=0).reshape(-1, 1)
    print(x_mean.shape)
    mu_pos = np.matmul(
        cov_pos,
        np.matmul(inv_sigma0, mu0.reshape(-1, 1)) + X.shape[0] * np.matmul(inv_sigma, x_mean)
    )

    return mu_pos, cov_pos
