import torch
import torch.nn as nn

class VRNN(nn.Module):

    def __init__(self, params):
        super(VRNN, self).__init__()

        self.h_dim = params['h_dim']
        self.z_dim = params['z_dim']
        self.x_dim = params['x_dim']

        self.x_feats = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )

        self.prior_feats = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )

        self.prior_mu = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
        )

        self.prior_sigma = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus()
        )

        self.enc_feats = nn.Sequential(
            nn.Linear(self.h_dim*2, self.h_dim),
            nn.ReLU()
        )

        self.enc_mu = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
        )

        self.enc_sigma = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus()
        )

        self.z_feats = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            nn.Linear(2*self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.x_dim),
        )

        self.recurrence = nn.GRUCell(2*self.h_dim, self.h_dim)

    def forward(self, x):
        """
        Parameters:
        - x (tensor): shape (sequence x batch x features)

        Returns:
        - p_mus (tensor): prior mean of shape
                          (batch x sequence x h_dim)
        - p_sigmas (tensor): prior std.dev. of shape
                             (batch x sequence x h_dim)
        - z_mus (tensor): posterior mean of shape
                          (batch x sequence x h_dim)
        - z_sigmas (tensor): posterior std.dev. of shape
                             (batch x sequence x h_dim)
        - decoded (tensor): decoded features of shape
                            (batch x sequence x features)

        """

        h = torch.zeros([x.shape[0], self.h_dim], device=x.device)
        p_mus = []; p_sigmas = []; z_mus = []; z_sigmas = [];
        decoded = [];

        for i in range(x.shape[1]):
            # feature extractor
            x_feat = self.x_feats(x[:, i])

            # prior
            p_feat = self.prior_feats(h)
            p_mu = self.prior_mu(p_feat)
            p_sigma = self.prior_sigma(p_feat)

            # inference
            post_feat = self.enc_feats(torch.cat([x_feat, h], dim=1))
            z_mu = self.enc_mu(post_feat)
            z_sigma = self.enc_sigma(post_feat)

            # generation
            z_sample = self.reparameterise(z_mu, z_sigma)
            z_feat = self.z_feats(z_sample)
            dec = self.dec(torch.cat([z_feat, h], dim=1))

            # recurrence
            h = self.recurrence(torch.cat([x_feat, z_feat], dim=1))

            p_mus.append(p_mu)
            p_sigmas.append(p_sigma)
            z_mus.append(z_mu)
            z_sigmas.append(z_sigma)
            decoded.append(dec)

        data = [p_mus, p_sigmas, z_mus, z_sigmas, decoded]

        return [torch.stack(i, axis=1) for i in data]

    @staticmethod
    def reparameterise(z_mean, z_sigma):
        eps = torch.rand_like(z_mean, device=z_mean.device)
        return z_mean + z_sigma * eps
