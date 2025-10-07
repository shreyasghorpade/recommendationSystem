import sqlite3
import os
import json
from datetime import datetime

# ---------------- Database Path ----------------
DB_PATH = os.path.join(os.path.dirname(__file__), "../database/app.db")

# ---------------- Utility: Connect ----------------
def get_connection():
    """Returns a new DB connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # enables dict-style access
    return conn



# ---------------- User Functions ----------------
def add_user(username, password, first_name, last_name, email, favorites=None):
    """Adds a new user to the database."""
    conn = get_connection()
    cur = conn.cursor()
    fav_json = json.dumps(favorites or [])
    cur.execute("""
        INSERT INTO users (username, password, first_name, last_name, email, favorites, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, password, first_name, last_name, email, fav_json, datetime.now()))
    conn.commit()
    conn.close()


def get_user(username):
    """Fetch a user record by username."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------- Favorite Management ----------------
def add_favorite(user_id, movie_id):
    """Add a movie to user's favorites."""
    conn = get_connection()
    cur = conn.cursor()

    # Fetch existing favorites
    cur.execute("SELECT favorites FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise ValueError("User not found")

    favorites = json.loads(row["favorites"] or "[]")
    if movie_id not in favorites:
        favorites.append(movie_id)

    cur.execute("UPDATE users SET favorites = ? WHERE id = ?", (json.dumps(favorites), user_id))
    conn.commit()
    conn.close()


def remove_favorite(user_id, movie_id):
    """Remove a movie from user's favorites."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT favorites FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()

    if not row:
        conn.close()
        raise ValueError("User not found")

    favorites = json.loads(row["favorites"] or "[]")
    if movie_id in favorites:
        favorites.remove(movie_id)

    cur.execute("UPDATE users SET favorites = ? WHERE id = ?", (json.dumps(favorites), user_id))
    conn.commit()
    conn.close()


def get_user_favorites(user_id):
    """Return a list of movie IDs from user's favorites."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT favorites FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row or not row["favorites"]:
        return []
    return json.loads(row["favorites"])


# ---------------- Movie Access ----------------
def get_movie(movie_id):
    """Fetch movie details by its reindexed ID."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM movies WHERE id = ?", (movie_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_movies(limit=20):
    """Fetch a limited list of movies."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM movies LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Delete feature not added in the website
def delete_last_user():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id = (SELECT MAX(id) FROM users)")
    conn.commit()
    conn.close()
    print("✅ Last user deleted successfully!")
    
    
delete_last_user()