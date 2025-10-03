import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------Matrix Factorization Model-----------------
class MF_Implicit(nn.Module):
    def __init__(self, n_users, n_items, k=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        # dot product
        return (self.user_emb(u) * self.item_emb(i)).sum(dim=1)
    
    
# --------------GMF model---------------------
class GMF_Implicit(nn.Module):
    def __init__(self, n_users, n_items, k=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # linear layer on top of elementwise product
        self.fc = nn.Linear(k, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, u, i):
        z = self.user_emb(u) * self.item_emb(i)  # elementwise product
        return self.fc(z).view(-1) 
    

# --------------- NeuMF Model -----------
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, k_gmf=32, k_mlp=32, mlp_layers=(64,32,16)):
        super().__init__()
        self.ug = nn.Embedding(n_users, k_gmf); nn.init.normal_(self.ug.weight, std=0.01)
        self.ig = nn.Embedding(n_items, k_gmf); nn.init.normal_(self.ig.weight, std=0.01)
        self.um = nn.Embedding(n_users, k_mlp); nn.init.normal_(self.um.weight, std=0.01)
        self.im = nn.Embedding(n_items, k_mlp); nn.init.normal_(self.im.weight, std=0.01)
        layers = []; d = 2*k_mlp
        for h in mlp_layers:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(k_gmf + mlp_layers[-1], 1)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, u, i):
        g = self.ug(u) * self.ig(i)
        m = self.mlp(torch.cat([self.um(u), self.im(i)], dim=1))
        return self.fc(torch.cat([g, m], dim=1)).view(-1)