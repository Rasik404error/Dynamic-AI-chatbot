import sqlite3
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "database" / "chatbot.db"

DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


def get_connection():

    return sqlite3.connect(DATABASE_PATH)


def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS interactions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,

                user_message TEXT NOT NULL,

                assistant_response TEXT NOT NULL,

                sentiment TEXT,

                emotion TEXT,

                response_time REAL,

                feedback INTEGER
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS preferences (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                preference TEXT UNIQUE NOT NULL
            )
        """)


def save_interaction(
    user_message,
    assistant_response,
    sentiment,
    emotion,
    response_time
):

    with get_connection() as connection:

        cursor = connection.execute("""
            INSERT INTO interactions (
                timestamp,
                user_message,
                assistant_response,
                sentiment,
                emotion,
                response_time
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            user_message,
            assistant_response,
            sentiment,
            emotion,
            response_time
        ))

        return cursor.lastrowid


def save_feedback(interaction_id, feedback):

    with get_connection() as connection:

        connection.execute("""
            UPDATE interactions
            SET feedback = ?
            WHERE id = ?
        """, (
            feedback,
            interaction_id
        ))


def get_preferences():

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT preference
            FROM preferences
            ORDER BY id DESC
            LIMIT 10
        """).fetchall()

    return [row[0] for row in rows]


def add_preference(preference):

    with get_connection() as connection:

        connection.execute("""
            INSERT OR IGNORE INTO preferences (preference)
            VALUES (?)
        """, (preference,))


def get_interactions():
    import pandas as pd

    with get_connection() as connection:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM interactions
            ORDER BY timestamp DESC
            """,
            connection
        )

    return df