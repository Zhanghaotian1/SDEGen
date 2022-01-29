import torch

def loss_fn_discretized(score, target, sigmas):
    if score.shape[1] == 1 :
        score = score.view(-1)
        target = target.view(-1)
        sigmas = sigmas.view(-1)
    loss = 0.5 *((score - target) **2 ) * sigmas
    return torch.mean(loss)