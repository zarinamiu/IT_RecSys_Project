import os
import sqlite3
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query

DB_NAME = "recsys.db"
CSV_PATH = "/Users/zarinagimazova/PycharmProjects/IT_RecSys_Project/IT_RecSys_Project/udemy_courses.csv"


def init_database():
    if os.path.exists(DB_NAME):
        return

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Файл {CSV_PATH} не найден.")

    df = pd.read_csv(CSV_PATH)
    columns_to_keep = [
        'course_id', 'course_title', 'url', 'is_paid', 'price',
        'num_subscribers', 'num_reviews', 'num_lectures', 'level',
        'content_duration', 'published_timestamp', 'subject'
    ]

    courses_df = df[columns_to_keep].drop_duplicates(subset=['course_id'])
    courses_df['subject'] = courses_df['subject'].astype(str).str.strip()
    courses_df['course_title'] = courses_df['course_title'].astype(str).str.strip()

    all_course_ids = courses_df['course_id'].unique()
    total_courses = len(all_course_ids)
    ratings_records = []

    for user_idx in range(1, 101):
        user_id = f"user_{user_idx}"
        sample_size = min(total_courses, int(np.random.randint(15, 31)))
        chosen_courses = np.random.choice(all_course_ids, size=sample_size, replace=False)

        for course_id in chosen_courses:
            score = int(np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3]))
            ratings_records.append((user_id, int(course_id), score))

    ratings_df = pd.DataFrame(ratings_records, columns=['user_id', 'course_id', 'rating'])

    with sqlite3.connect(DB_NAME) as conn:
        courses_df.to_sql('items', conn, if_exists='replace', index=False, dtype={'course_id': 'INTEGER PRIMARY KEY'})
        ratings_df.to_sql('ratings', conn, if_exists='replace', index=False)

        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_uid ON ratings(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cid ON ratings(course_id);")
        conn.commit()


init_database()

app = FastAPI(title="RecSys API")


class RecEngine:
    @staticmethod
    def load_data():
        with sqlite3.connect(DB_NAME) as conn:
            items_df = pd.read_sql_query("SELECT * FROM items", conn)
            ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)

        items_df['course_id'] = items_df['course_id'].astype(int)
        ratings_df['course_id'] = ratings_df['course_id'].astype(int)
        ratings_df['user_id'] = ratings_df['user_id'].astype(str)
        return items_df, ratings_df

    @classmethod
    def get_popular(cls, n=3):
        items, _ = cls.load_data()
        top_items = items.sort_values(by=['num_subscribers', 'num_reviews'], ascending=False).head(n)
        return top_items[['course_id', 'course_title', 'subject']].to_dict(orient='records')

    @classmethod
    def get_by_category(cls, category, n=3):
        items, _ = cls.load_data()
        filtered = items[items['subject'].str.lower() == category.lower().strip()]
        top_by_cat = filtered.sort_values(by='num_subscribers', ascending=False).head(n)
        return top_by_cat[['course_id', 'course_title', 'subject']].to_dict(orient='records')

    @classmethod
    def get_content_based(cls, user_id, n=2):
        items, ratings = cls.load_data()
        user_good = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
        if user_good.empty:
            return pd.DataFrame()

        merged = user_good.merge(items, on='course_id')
        fav_categories = merged['subject'].value_counts().index.tolist()
        watched = ratings[ratings['user_id'] == user_id]['course_id'].tolist()

        candidates = items[items['subject'].isin(fav_categories) & (~items['course_id'].isin(watched))]
        if not candidates.empty:
            candidates = candidates.sample(frac=1).reset_index(drop=True)
        return candidates.head(n)

    @classmethod
    def get_collaborative(cls, user_id, n=2):
        items, ratings = cls.load_data()
        if ratings.empty or user_id not in ratings['user_id'].values:
            return pd.DataFrame()

        user_history = ratings[ratings['user_id'] == user_id]['course_id'].unique()
        similar_users = ratings[ratings['course_id'].isin(user_history)]['user_id'].unique()

        filtered_matrix = ratings[ratings['user_id'].isin(similar_users)]
        matrix = filtered_matrix.pivot(index='user_id', columns='course_id', values='rating').fillna(0)

        if user_id not in matrix.index:
            return pd.DataFrame()

        vectors = matrix.values
        target_idx = matrix.index.get_loc(user_id)
        target_vector = vectors[target_idx]

        norms = np.linalg.norm(vectors, axis=1)
        target_norm = norms[target_idx]

        if target_norm == 0:
            return pd.DataFrame()

        with np.errstate(invalid='ignore', divide='ignore'):
            dots = np.dot(vectors, target_vector)
            sims = dots / (norms * target_norm)
            sims = np.nan_to_num(sims)

        sim_series = pd.Series(sims, index=matrix.index).drop(user_id)
        top_users = sim_series.nlargest(3).index.tolist()

        seen = ratings[ratings['user_id'] == user_id]['course_id'].tolist()
        rec_filter = (
                ratings['user_id'].isin(top_users) &
                (ratings['rating'] >= 4) &
                (~ratings['course_id'].isin(seen))
        )
        rec_ids = ratings[rec_filter]['course_id'].unique()
        return items[items['course_id'].isin(rec_ids[:n])]


@app.get("/recommendations/general")
def general_recommendations():
    return {
        "type": "popular",
        "data": RecEngine.get_popular(n=3)
    }


@app.get("/recommendations/personalized")
def personalized_recommendations(
        user_id: str = Query(None),
        subject: str = Query(None)
):
    _, ratings = RecEngine.load_data()

    if subject and (not user_id or user_id.strip() not in ratings['user_id'].values):
        cat_data = RecEngine.get_by_category(category=subject, n=3)
        if not cat_data:
            cat_data = RecEngine.get_popular(n=3)
        return {
            "type": "cold_start_category",
            "subject": subject,
            "data": cat_data
        }

    if user_id:
        clean_uid = user_id.strip()
        if clean_uid in ratings['user_id'].values:
            c_recs = RecEngine.get_content_based(clean_uid, n=2)
            col_recs = RecEngine.get_collaborative(clean_uid, n=2)

            combined = pd.concat([c_recs, col_recs]).drop_duplicates(subset=['course_id'])

            if not combined.empty:
                res = combined.head(3)[['course_id', 'course_title', 'subject']].to_dict(orient='records')
                return {
                    "type": "personalized",
                    "user_id": clean_uid,
                    "data": res
                }

    return {
        "type": "fallback_popular",
        "data": RecEngine.get_popular(n=3)
    }


if __name__ == "__main__":
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{current_file}:app", host="127.0.0.1", port=8000, reload=True)
