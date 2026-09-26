"""Use shared viewing preferences for existing users and movie text for new users."""

import csv
import gzip
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_csv(path):
    """Read an ordinary or gzip-compressed CSV into dictionaries."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def fit_item_neighbors(histories, movie_count, n_neighbors):
    """Find movies with similar preferences across at least two shared users."""
    movie_indices, user_indices, weights = [], [], []
    for user_index, history in enumerate(histories.values()):
        for movie_index, weight in history.items():
            movie_indices.append(movie_index)
            user_indices.append(user_index)
            weights.append(weight)

    # Rows are movies and columns are users. Missing interactions stay zero.
    interactions = csr_matrix(
        (weights, (movie_indices, user_indices)),
        shape=(movie_count, len(histories)),
        dtype=np.float32,
    )
    if not histories:
        return csr_matrix((movie_count, movie_count), dtype=np.float32)

    similarities = cosine_similarity(interactions, dense_output=False)
    # One shared user is too little evidence for a movie-to-movie connection.
    observed = interactions.copy()
    observed.data = np.ones(len(observed.data), dtype=np.int32)
    shared_users = observed @ observed.T
    similarities = similarities.multiply(shared_users >= 2).tocsr()
    similarities.setdiag(0)  # A movie cannot be its own neighbor.
    # Tiny positive values can be floating-point noise rather than a connection.
    similarities.data[similarities.data <= 1e-6] = 0
    similarities.eliminate_zeros()

    neighbor_indices, neighbor_weights, row_starts = [], [], [0]
    for movie_index in range(movie_count):
        start, end = similarities.indptr[movie_index:movie_index + 2]
        indices = similarities.indices[start:end]
        values = similarities.data[start:end]
        # Movie indices follow sorted movie IDs, making equal scores repeatable.
        best = np.lexsort((indices, -values))[:n_neighbors]
        neighbor_indices.extend(indices[best])
        neighbor_weights.extend(values[best])
        row_starts.append(len(neighbor_indices))

    return csr_matrix(
        (neighbor_weights, neighbor_indices, row_starts),
        shape=(movie_count, movie_count),
        dtype=np.float32,
    )


def train_model(data_dir, n_neighbors=20):
    """Learn movie neighbors from interactions and word weights for cold start."""
    if not isinstance(n_neighbors, int) or n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer.")
    data_dir = Path(data_dir)

    def data_path(name):
        compressed = data_dir / f"{name}.csv.gz"
        return compressed if compressed.exists() else data_dir / f"{name}.csv"

    rows = read_csv(data_path("movies"))
    movies = sorted(
        [{key: row[key] for key in ("movie_id", "title", "genres")} for row in rows],
        key=lambda movie: movie["movie_id"],
    )
    movie_index = {movie["movie_id"]: index for index, movie in enumerate(movies)}
    if not movies or len(movie_index) != len(movies):
        raise ValueError("movies.csv must contain movies with unique movie_id values.")

    # Repeating genres makes them more influential than a single plot word.
    rows.sort(key=lambda row: row["movie_id"])
    descriptions = [
        " ".join([row["title"], *([row["genres"]] * 3), row.get("overview", "")])
        for row in rows
    ]
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=10_000, dtype=np.float32
    )
    movie_vectors = vectorizer.fit_transform(descriptions)

    histories = {}
    latest_ratings = {}
    for event in read_csv(data_path("events")):
        if event["event_type"] not in ("watch", "rating"):
            continue
        user_id, movie_id = event["user_id"], event["movie_id"]
        if movie_id not in movie_index:
            raise ValueError(f"Event references a movie missing from movies.csv: {movie_id}")
        index = movie_index[movie_id]
        history = histories.setdefault(user_id, {})
        if event["event_type"] == "watch":
            # A minute of video is weak evidence. Repeated minutes do not add up.
            history.setdefault(index, 0.25)
            continue

        try:
            rating = float(event["rating"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid rating for user {user_id}, movie {movie_id}.") from error
        if not np.isfinite(rating) or not 1 <= rating <= 10:
            raise ValueError(f"Rating must be between 1 and 10: {event['rating']!r}")
        key = (user_id, index)
        # The dataset uses ISO timestamps, so alphabetical order is time order.
        if key not in latest_ratings or event["timestamp"] >= latest_ratings[key]:
            latest_ratings[key] = event["timestamp"]
            history[index] = (rating - 5.5) / 4.5

    # Count people, not streamed minutes. Used only for fallback and score ties.
    popularity = np.zeros(len(movies), dtype=np.int64)
    for history in histories.values():
        for index in history:
            popularity[index] += 1

    return {
        "model_type": "item_cf_with_content_cold_start",
        "n_neighbors": n_neighbors,
        "item_neighbors": fit_item_neighbors(histories, len(movies), n_neighbors),
        "vectorizer": vectorizer,
        "movie_vectors": movie_vectors,
        "movies": movies,
        "histories": histories,
        "popularity": popularity,
    }


def recommend(model, user_id, top_k=20, profile=None):
    """Rank unseen movies using collaborative filtering or a new user's profile."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    movie_vectors = model["movie_vectors"]
    history = model["histories"].get(str(user_id), {})
    if history:
        indices = list(history)
        weights = np.array(list(history.values()), dtype=np.float32)
        # Each candidate uses only its neighbors that this user has seen.
        neighbors = model["item_neighbors"][:, indices]
        totals = np.asarray(neighbors @ weights).ravel()
        similarity_sums = np.asarray(neighbors.sum(axis=1)).ravel()
        scores = np.divide(
            totals,
            similarity_sums,
            out=np.zeros(len(model["movies"]), dtype=np.float32),
            where=similarity_sums > 0,
        )
    else:
        if profile:
            # Transform separately so disliked words lower a movie's score.
            preferences = model["vectorizer"].transform(
                [profile.get("likes", ""), profile.get("dislikes", "")]
            )
            user_vector = (preferences[0] - preferences[1]).toarray().ravel()
        else:
            user_vector = np.zeros(movie_vectors.shape[1], dtype=np.float32)

        length = np.linalg.norm(user_vector)
        if length > 0:
            user_vector = user_vector / length
        # TF-IDF normalizes movie vectors, so this is cosine similarity.
        scores = np.asarray(movie_vectors @ user_vector).ravel()
    movies = model["movies"]
    ranked = sorted(
        (index for index in range(len(movies)) if index not in history),
        key=lambda index: (
            -float(scores[index]),
            -int(model["popularity"][index]),
            movies[index]["movie_id"],
        ),
    )
    return [dict(movies[index], score=float(scores[index])) for index in ranked[:top_k]]
