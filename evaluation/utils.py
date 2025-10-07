import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------Preparing dataset----------------------------
# ----------------Preparing dataset (for reindexed data)----------------------------
def prepare_dataset(path):
    """
    Load already reindexed ratings data and split into train/test.
    Expects columns: userId, movieId, rating, label(optional)
    """
    ratings = pd.read_csv(path)

    # Ensure label column exists
    if "label" not in ratings.columns:
        ratings["label"] = (ratings["rating"] >= 4.0).astype(int)

    n_users = ratings["userId"].nunique()
    n_items = ratings["movieId"].nunique()

    # One positive per user for test
    pos = ratings[ratings.label == 1]
    test_idx = pos.groupby("userId", group_keys=False).apply(
        lambda x: x.sample(1, random_state=42)
    ).index
    test_df = ratings.loc[test_idx][["userId", "movieId", "label"]]
    train_df = ratings.drop(test_idx)

    train_ui = set(zip(train_df.userId.tolist(), train_df.movieId.tolist()))

    return n_users, n_items, test_df, train_df, train_ui

    
    
# ----------------Negative Sampling for training---------------------
def sample_train_batch(train_df, n_items, train_ui, num_neg=4, users_subset=None):
    if users_subset is None:
        users = train_df[train_df.label==1]["userId"].unique().tolist()
    else:
        users = users_subset
    U,I,Y = [],[],[]
    for u in users:
        pos_items = train_df[(train_df.userId==u) & (train_df.label==1)].movieId.tolist()
        if not pos_items: continue
        ip = random.choice(pos_items)
        U.append(u); I.append(ip); Y.append(1.0)
        negs=0
        while negs<num_neg:
            j = random.randint(0, n_items-1)
            if (u,j) not in train_ui and j!=ip:
                U.append(u); I.append(j); Y.append(0.0)
                negs+=1
    return torch.LongTensor(U), torch.LongTensor(I), torch.FloatTensor(Y)