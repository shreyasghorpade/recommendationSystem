import sqlite3
import pandas as pd
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "../database/app.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- Drop old tables ---
    cur.executescript("""
    DROP TABLE IF EXISTS users;
    DROP TABLE IF EXISTS movies;
    """)

    # --- Create tables ---
    cur.executescript("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT,
        first_name TEXT,
        last_name TEXT,
        email TEXT,
        password TEXT,
        favorites TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE movies (
        id INTEGER PRIMARY KEY,
        title TEXT,
        genres TEXT
    );
    """)

    print("✅ Tables created successfully!")

    # --- Insert reindexed movies ---
    movies = pd.read_csv("../data/rev_movieids.csv")
    for _, row in movies.iterrows():
        cur.execute("INSERT INTO movies (id, title, genres) VALUES (?, ?, ?)",
                    (int(row["reindexed_movieId"]), row["title"], row["genres"]))
    print(f"🎬 Inserted {len(movies)} movies.")

    # --- Insert users from reindexed ratings ---
    ratings = pd.read_csv("../data/ratings_reindexed.csv")
    user_ids = ratings["userId"].unique()
    for uid in user_ids:
        username = f"user_{uid}"
        cur.execute("""
            INSERT INTO users (id, username, first_name, last_name, email, password, favorites)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (int(uid), username, "Default", "User",
              f"{username}@example.com", "password123", json.dumps([])))
    print(f"👤 Inserted {len(user_ids)} users.")

    conn.commit()
    conn.close()
    print("💾 Database initialized successfully!")

if __name__ == "__main__":
    init_db()
