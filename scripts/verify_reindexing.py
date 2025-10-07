# scripts/verify_mappings.py
import json, pandas as pd

# Load mappings
with open("../mappings/id_mappings.json") as f:
    mappings = json.load(f)

uid2idx = {int(k): v for k, v in mappings["uid2idx"].items()}
iid2idx = {int(k): v for k, v in mappings["iid2idx"].items()}

# Load reindexed data
ratings = pd.read_csv("../data/ratings_reindexed.csv")
movies = pd.read_csv("../data/rev_movieids.csv")

print(f"🔍 Ratings: {len(ratings)} rows | Users: {ratings['userId'].nunique()} | Movies: {ratings['movieId'].nunique()}")
print(f"🔍 Mapping JSON: {len(uid2idx)} users, {len(iid2idx)} movies")
print(f"🔍 Movies mapping CSV: {len(movies)} entries")

# Check alignment
missing_user = set(ratings["userId"]) - set(uid2idx.values())
missing_movie = set(ratings["movieId"]) - set(iid2idx.values())

if not missing_user and not missing_movie:
    print("✅ All reindexed IDs perfectly match mappings.")
else:
    if missing_user: print(f"❌ Missing users: {len(missing_user)}")
    if missing_movie: print(f"❌ Missing movies: {len(missing_movie)}")
