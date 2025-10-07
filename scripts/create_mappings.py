import os, json
import pandas as pd

# ---------- STEP 1: Load original data ----------
ratings = pd.read_csv("../data/ratings.csv").drop(columns=["timestamp"], errors="ignore")
movies = pd.read_csv("../data/movies.csv")

# Add label column for implicit feedback
ratings["label"] = (ratings["rating"] >= 4.0).astype(int)

# ---------- STEP 2: Reindex users and items ----------
uids = ratings["userId"].unique()
iids = ratings["movieId"].unique()

n_users = len(uids)
n_items = len(iids)

# Same logic as previous prepare_dataset()
uid2idx = {u: i for i, u in enumerate(uids)}
iid2idx = {m: i for i, m in enumerate(iids)}

revUserIdx = {v: k for k, v in uid2idx.items()}
revItemIdx = {v: k for k, v in iid2idx.items()}

# Map userId and movieId to new indices
ratings["userId"] = ratings["userId"].map(uid2idx)
ratings["movieId"] = ratings["movieId"].map(iid2idx)

# Drop any rows that became NaN (shouldn’t happen)
ratings = ratings.dropna(subset=["userId", "movieId"])
ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

print(f"✅ Reindexed {n_users} users and {n_items} movies.")

# ---------- STEP 3: Save id mappings ----------
os.makedirs("../mappings", exist_ok=True)

uid2idx_str = {str(k): int(v) for k, v in uid2idx.items()}
iid2idx_str = {str(k): int(v) for k, v in iid2idx.items()}

mappings = {
    "uid2idx": uid2idx_str,
    "iid2idx": iid2idx_str,
}

with open("../mappings/id_mappings.json", "w") as f:
    json.dump(mappings, f, indent=4)

print("💾 Saved uid2idx & iid2idx → mappings/id_mappings.json")

# ---------- STEP 4: Create reverse movie mapping ----------
movies = movies[movies["movieId"].isin(iids)]
rev_movieids = pd.DataFrame({
    "original_movieId": list(iids),
    "reindexed_movieId": [iid2idx[m] for m in iids],
})
rev_movieids = rev_movieids.merge(
    movies, left_on="original_movieId", right_on="movieId", how="left"
).drop(columns=["movieId"])

rev_movieids.to_csv("../data/rev_movieids.csv", index=False)
print("🎬 Saved reindexed movie mapping → data/rev_movieids.csv")

# ---------- STEP 5: Save reindexed ratings ----------
ratings.to_csv("../data/ratings_reindexed.csv", index=False)
print("💾 Saved reindexed ratings → data/ratings_reindexed.csv")

# ---------- STEP 6: Verification ----------
print("\n🔍 Verification:")
print(f"  Unique users (mapping): {len(uid2idx)} | in CSV: {ratings['userId'].nunique()}")
print(f"  Unique movies (mapping): {len(iid2idx)} | in CSV: {ratings['movieId'].nunique()}")
print(f"  Total ratings: {len(ratings)}")
print("✅ All mappings created successfully!")
