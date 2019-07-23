#!/usr/bin/env python

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
import utils


class ScoreFunctionEstimatorBBVI:
    """
    The most basic black box variational inference using the score function estimator.

    In this example, we assume that p(z) and p(x|z) are both Gaussian-distributed.
    Specifically, we assume that p(x|z) has a known covariance matrix Sigma, the mean
    vector of x is z, and z has a prior distribution of N(\mu_0, \Sigma_0).

    Therefore, by conjugacy, the posterior p(z|x) should be a Gaussian distribution as well.

    The variational distribution q(z) is assumed to be a factored Gaussian.
    """
    def __init__(self, dim=2, sample_size=100, data_size=100, seed=42):
        self.dim = dim
        self.monte_carlo_sample_size = sample_size
        self.data_size = data_size

        np.random.seed(seed)

        # parameters for q(z), which are to be estimated
        # (mean-field Gaussian)
        self.z_mu = [torch.zeros(1, requires_grad=True) for _ in range(dim)]
        self.z_sigma = [torch.ones(1, requires_grad=True) for _ in range(dim)]

        # parameter for the prior: MVN(0, I)
        self.z_mu0 = torch.from_numpy(np.zeros((dim, ), dtype=np.float32))
        self.z_sigma0 = torch.from_numpy(np.eye(dim, dtype=np.float32) * 5)

        self.z_distn = MultivariateNormal(self.z_mu0, self.z_sigma0)

        # likelihood parameters (covariance matrix)
        # create a pos-def matrix by generating a random matrix A, then compute A'A.
        A = (np.random.random((dim, dim)).astype(np.float32) * 2 - 1) * 4
        self.x_cov = torch.from_numpy(np.matmul(A.T, A))

        self.X = self.generate_x(sample_shape=(self.data_size, ))

    def draw_from_q(self, n=None):
        """
        Draw `n` random samples z_i from the variational distribution q(z).
        In this case, we assume that q(z) factorizes into:

           q(z) = \prod_i q(z_i|\mu_i, \sigma_i)

        :return:
        """
        if n is None:
            n = self.monte_carlo_sample_size

        sampled_Z = np.empty((n, self.dim), dtype=np.float32)

        for d in range(self.dim):
            _mu = self.z_mu[d].item()
            _sigma = self.z_sigma[d].item()
            sampled_Z[:, d] = np.random.normal(_mu, _sigma, (n,)).astype(np.float32)

        return sampled_Z

    def generate_x(self, sample_shape=None):
        """
        Generate a random x_i vector from the likelihood function by:

         1. generate z' from the prior p(z)
         2. generate x from the likelihood, which has distribution N(z', Sigma), where Sigma is known.

        This way we don't have to "provide" any data. We can just generate random data points
        on the fly, so to speak.
        :return:
        """
        # draw z from the prior
        z_gen = self.z_distn.sample()
        x_distn = MultivariateNormal(z_gen, self.x_cov)

        if sample_shape is None:
            return x_distn.sample()
        else:
            # sample_shape must be iterable
            if isinstance(sample_shape, int):
                sample_shape = (sample_shape, )

            return x_distn.sample(sample_shape)

    def plot_posterior(self, **kwargs):
        mu_pos, cov_pos = utils.mvn_posterior(
            self.z_mu0.numpy(), self.z_sigma0.numpy(),
            self.x_cov.numpy(),
            self.X.numpy()
        )
        print(f"Posterior mean: {mu_pos}")
        utils.contour_plot_mvn(mu_pos, cov_pos, **kwargs)

    def compute_joint_log_prob(self, X, Z):
        """
        Compute the logarithm of the PDF of the joint distribution p(x, z).
        :param x: an input data vector (x_i).
        :param Z: sampled z vectors (from the variational distribution).
        :return:
        """
        # prior
        prior_log_pdf = self.z_distn.log_prob(torch.from_numpy(Z))
        likelihood_log_pdf = np.zeros(Z.shape[0], dtype=np.float32)

        # compute the likelihood
        for i in range(Z.shape[0]):
            multinorm = MultivariateNormal(torch.from_numpy(Z[i, :]), self.x_cov)
            log_pdf = multinorm.log_prob(X)
            likelihood_log_pdf[i] = torch.sum(log_pdf).item()

        # compute the joint probability
        return prior_log_pdf + torch.tensor(likelihood_log_pdf)

    def compute_variational_log_prob(self, Z):
        # compute the joint probabilities for the factored Gaussian VI distribution.
        joint_log_prob = torch.zeros(Z.shape[0])
        for i in range(self.dim):
            q_i_distn = Normal(self.z_mu[i], self.z_sigma[i])
            coord_log_prob = q_i_distn.log_prob(torch.from_numpy(Z[:, i]))
            # multiplying the factored probabilities (add since we are on log-scale.)
            joint_log_prob += coord_log_prob

        return joint_log_prob

    def loss_fn(self, X, Z):
        """
        Specify the loss-function, which is the negative evidence lower bound (ELBO), since
        PyTorch's optimizer always wants to minimize the loss, whereas we'd like to maximize the
        ELBO.
        :return:
        """
        joint_log_prob = self.compute_joint_log_prob(X, Z)

        # log probability of the variational dist'n.
        # we don't need to compute gradient for this part, hence the torch.no_grad().
        with torch.no_grad():
            log_qz_prob = self.compute_variational_log_prob(Z)
            assert not log_qz_prob.requires_grad

        # the log[q(z)] term from which we need to compute the gradient.
        log_qz_prob_with_grad = self.compute_variational_log_prob(Z)

        # we treat (joint_log_prob - log_qz_prob) as normalizing constants, since we don't need
        # to compute gradients from these two terms.
        return -torch.mean((joint_log_prob - log_qz_prob) * log_qz_prob_with_grad)

    def fit(self, epochs=10):
        optimizer = torch.optim.Adam(
            # trying estimate variational parameters for q(z)
            params=self.z_mu + self.z_sigma,
            lr=0.03
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)
        for epoch in range(epochs):
            optimizer.zero_grad()

            # draw samples of z from q(z), and compute loss using Monte Carlo
            Z = self.draw_from_q()
            loss = self.loss_fn(self.X, Z)

            print(f"Iteration {epoch:4d}: loss = {loss.item()}")

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                fig = plt.gcf()
                utils.contour_plot_mvn(
                    np.array([m.item() for m in self.z_mu]),
                    np.diag([s.item() for s in self.z_sigma]) ** 2,
                    cmap=cm.Blues, alpha=0.5
                )
                self.plot_posterior(cmap=cm.Reds, alpha=0.5)

            plt.show()
            print(f"q_mu: {[m.item() for m in self.z_mu]}")
            print(f"q_sigma: {[s.item() for s in self.z_sigma]}")


def main():
    bbvi = ScoreFunctionEstimatorBBVI(data_size=50, sample_size=200)
    fig = plt.gcf()
    bbvi.plot_posterior()
    plt.scatter(bbvi.X[:, 0], bbvi.X[:, 1])
    plt.show()
    bbvi.fit(2500)


if __name__ == "__main__":
    main()