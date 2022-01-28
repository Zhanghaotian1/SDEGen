import torch


def loss_fn(model, data, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    node2graph = data.batch
    edge2graph = node2graph[data.edge_index[0]]
    graph_num = len(data.smiles)
    d = data.edge_length  # (num_edge,1)
    random_t = torch.rand(graph_num, device=d.device) * (1. - eps) + eps  # (batch_size)
    z = torch.randn_like(d)  # (num_edge,1)
    std = marginal_prob_std(random_t)[edge2graph]  # (num_edge)
    perturbed_d = d + z * std[:, None]  # std[:,None]转化尺度为(num_edge,1)，perturbed_d.size() = (edge_num,1)
    data.edge_length = perturbed_d
    score = model(data, random_t)
    loss = torch.mean((score[:, None] * std[:, None] + z) ** 2)
    return loss