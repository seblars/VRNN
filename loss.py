import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F
import torch

def loss_function(data, x, params):

    prior_mu, prior_sigma, z_mu, z_sigma, x_decoded = data

    # manifold loss
    norm_p = Norm.Normal(prior_mu, prior_sigma)
    norm_z = Norm.Normal(z_mu, z_sigma)
    kl_loss = torch.mean(KL.kl_divergence(norm_z, norm_p), dim=[1,2]).sum()

    # # reconstruction loss
    rc_loss = torch.mean(F.mse_loss(
                                    x_decoded, x,
                                    reduction='none'),
                         dim=[1,2]).sum()

    loss = (1 - params["kl_weight"]) * rc_loss + params["kl_weight"] * kl_loss

    return loss, rc_loss, kl_loss
