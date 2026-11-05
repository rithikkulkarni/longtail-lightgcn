"""LightGCN: parameter-free propagation over the normalized bipartite graph (He et al., 2020)."""
import torch
from torch import nn


class LightGCN(nn.Module):
    """Learnable parameters are only the layer-0 (ID) embeddings of users and items.

    Propagation:    E^(k+1) = A_hat @ E^(k)       (no weight matrices, no nonlinearity)
    Final embedding: mean of E^(0), ..., E^(K)    (layer weights 1/(K+1), as in the authors' code)
    Score:           dot product of final user and item embeddings.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        init_std: float = 0.01,
        item_features: torch.Tensor | None = None,
        mlp_hidden: int = 64,
    ):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=init_std)
        nn.init.normal_(self.item_emb.weight, std=init_std)

        # Intervention C (D-28): e_i^(0) = e_i^ID + f_theta(x_i). Without features no MLP is created, so the
        # model (and its random init) is exactly the baseline. The MLP is built after the embeddings above.
        self.content_mlp = None
        if item_features is not None:
            if item_features.shape[0] != n_items:
                raise ValueError(f"item_features has {item_features.shape[0]} rows, expected {n_items}")
            self.register_buffer("item_features", item_features.float(), persistent=False)
            self.content_mlp = nn.Sequential(
                nn.Linear(item_features.shape[1], mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, embed_dim)
            )

    def ego_embeddings(self) -> torch.Tensor:
        """Layer-0 embeddings stacked as (U+I, d); users first, items after, matching the adjacency.

        With Intervention C the item rows are the ID embedding plus the MLP of the item's content features.
        """
        items = self.item_emb.weight
        if self.content_mlp is not None:
            items = items + self.content_mlp(self.item_features)
        return torch.cat([self.user_emb.weight, items], dim=0)

    def propagate(self, adj: torch.Tensor):
        """Run K propagation layers and return (user_final, item_final)."""
        e = self.ego_embeddings()
        layers = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(adj, e)
            layers.append(e)
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]

    @staticmethod
    def score(user_final: torch.Tensor, item_final: torch.Tensor, users: torch.Tensor, items: torch.Tensor):
        """Dot-product score for each (user, item) pair in the batch."""
        return (user_final[users] * item_final[items]).sum(dim=-1)
