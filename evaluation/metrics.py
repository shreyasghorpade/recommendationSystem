import torch
import torch.nn as nn
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Evaluation (HR@K, NDCG@K)
def eval_hr_ndcg(model, test_df, train_ui, n_items, k=10, n_neg=100, device="cpu"):
    model.eval(); HR=0.0; N=0.0; cnt=0
    rng = np.random.default_rng(123)
    with torch.no_grad():
        for _, row in test_df.iterrows():
            u, ip = int(row.userId), int(row.movieId)
            seen = {i for (uu,i) in train_ui if uu==u}
            negs = []
            while len(negs)<n_neg:
                j = int(rng.integers(0,n_items))
                if j!=ip and j not in seen: negs.append(j)
            cands = [ip]+negs
            U = torch.LongTensor([u]*(1+n_neg)).to(device)
            I = torch.LongTensor(cands).to(device)
            scores = model(U,I).cpu().numpy()
            topk_idx = scores.argsort()[-k:][::-1]
            topk = [cands[t] for t in topk_idx]
            hit = 1.0 if ip in topk else 0.0; HR+=hit
            if hit:
                rank = topk.index(ip)+1
                N+=1.0/math.log2(rank+1)
            cnt+=1
    return HR/cnt, N/cnt