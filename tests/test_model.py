"""Small examples that check recommendation behavior without the real dataset."""

import csv
import gzip
from pathlib import Path
import tempfile
import unittest

import numpy as np

from model import read_csv, recommend, train_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_dir = Path(self.temp_dir.name)
        self.movies = [
            ["space", "Space Journey", "Science Fiction", "astronaut planet spaceship"],
            ["romance", "Love Story", "Romance", "love wedding relationship"],
            ["alien", "Alien Planet", "Science Fiction", "astronaut planet spaceship"],
            ["wedding", "Wedding Story", "Romance", "love wedding relationship"],
        ]
        self.write_csv(
            "movies.csv", ["movie_id", "title", "genres", "overview"], self.movies
        )
        # These users link two movies even though their text is unrelated.
        self.shared_ratings = [
            ["2025-01-01", "2", "rating", "space", "10"],
            ["2025-01-01", "2", "rating", "romance", "10"],
            ["2025-01-01", "3", "rating", "space", "10"],
            ["2025-01-01", "3", "rating", "romance", "10"],
        ]

    def write_csv(self, name, columns, rows):
        with (self.data_dir / name).open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(rows)

    def train(self, events, n_neighbors=20):
        self.write_csv(
            "events.csv", ["timestamp", "user_id", "event_type", "movie_id", "rating"], events
        )
        return train_model(self.data_dir, n_neighbors=n_neighbors)

    def test_collaborative_filtering_learns_from_shared_ratings_not_text(self):
        events = self.shared_ratings + [["2025-01-01", "1", "rating", "space", "10"]]
        model = self.train(events)
        rows = recommend(model, "1")
        self.assertEqual(rows[0]["movie_id"], "romance")
        self.assertAlmostEqual(rows[0]["score"], 1.0)
        self.assertEqual({row["movie_id"] for row in rows}, {"alien", "romance", "wedding"})
        self.assertEqual(model["model_type"], "item_cf_with_content_cold_start")

        # Changing every movie's text leaves the history-based ranking unchanged.
        changed_movies = [[row[0], "Different title", "Mystery", "detective"] for row in self.movies]
        self.write_csv(
            "movies.csv", ["movie_id", "title", "genres", "overview"], changed_movies
        )
        changed_rows = recommend(self.train(events), "1")
        self.assertEqual(
            [(row["movie_id"], row["score"]) for row in rows],
            [(row["movie_id"], row["score"]) for row in changed_rows],
        )

    def test_latest_rating_overrides_watch_minutes_and_earlier_rating(self):
        model = self.train([
            ["2025-01-01", "1", "watch", "space", ""],
            ["2025-01-03", "1", "rating", "space", "10"],
            ["2025-01-04", "1", "watch", "space", ""],
            ["2025-01-01", "1", "rating", "space", "1"],
            ["2025-01-05", "1", "watch", "space", ""],
            ["2025-01-05", "1", "watch", "romance", ""],
            ["2025-01-06", "1", "watch", "romance", ""],
        ])
        ids = [movie["movie_id"] for movie in model["movies"]]
        self.assertEqual(model["histories"]["1"][ids.index("space")], 1.0)
        self.assertEqual(model["histories"]["1"][ids.index("romance")], 0.25)
        self.assertEqual(model["popularity"][ids.index("space")], 1)
        recommendations = recommend(model, "1")
        self.assertEqual({row["movie_id"] for row in recommendations}, {"alien", "wedding"})

    def test_low_rating_reduces_scores_for_similar_movies(self):
        model = self.train(self.shared_ratings + [
            ["2025-01-01", "1", "rating", "space", "10"],
            ["2025-01-02", "1", "rating", "space", "1"],
        ])
        rows = recommend(model, "1")
        self.assertEqual(rows[-1]["movie_id"], "romance")
        self.assertAlmostEqual(rows[-1]["score"], -1.0)
        self.assertNotIn("space", {row["movie_id"] for row in rows})

    def test_score_averages_seen_neighbors_using_similarity(self):
        events = [
            ["2025-01-01", user, "rating", movie, "10"]
            for user in ("2", "3")
            for movie in ("space", "romance", "wedding")
        ]
        model = self.train(events + [
            ["2025-01-01", "1", "rating", "space", "10"],
            ["2025-01-01", "1", "rating", "wedding", "1"],
        ])
        rows = recommend(model, "1")
        # Romance has equally similar liked and disliked neighbors, which cancel.
        scores = {row["movie_id"]: row["score"] for row in rows}
        self.assertAlmostEqual(scores["romance"], 0.0)
        self.assertEqual(set(scores), {"alien", "romance"})

    def test_a_single_shared_user_is_not_enough_for_neighbors(self):
        model = self.train([
            ["2025-01-01", "1", "rating", "space", "10"],
            ["2025-01-01", "2", "rating", "space", "10"],
            ["2025-01-01", "2", "rating", "romance", "10"],
        ])
        self.assertEqual(model["item_neighbors"].nnz, 0)
        rows = recommend(model, "1")
        self.assertTrue(all(row["score"] == 0 for row in rows))
        self.assertEqual([row["movie_id"] for row in rows], ["romance", "alien", "wedding"])

    def test_neighbors_are_limited_and_ties_follow_movie_id(self):
        events = [
            ["2025-01-01", user, "rating", movie[0], "10"]
            for user in ("1", "2")
            for movie in self.movies
        ]
        model = self.train(events, n_neighbors=1)
        neighbors = model["item_neighbors"]
        self.assertEqual(model["n_neighbors"], 1)
        self.assertTrue(np.all(neighbors.diagonal() == 0))
        self.assertTrue(np.all(neighbors.getnnz(axis=1) == 1))
        self.assertTrue(np.all(neighbors.data > 0))
        ids = [movie["movie_id"] for movie in model["movies"]]
        for index, movie_id in enumerate(ids):
            expected = min(other for other in ids if other != movie_id)
            self.assertEqual(ids[neighbors[index].indices[0]], expected)
        self.assertEqual(recommend(model, "1"), [])

    def test_opposite_preferences_do_not_create_neighbors(self):
        model = self.train([
            ["2025-01-01", user, "rating", movie, rating]
            for user in ("1", "2")
            for movie, rating in (("space", "10"), ("romance", "1"))
        ])
        self.assertEqual(model["item_neighbors"].nnz, 0)

    def test_rounding_noise_does_not_create_a_movie_connection(self):
        # Preferences [1, 1/3, 1/3] and [1/3, -1/3, -2/3] have dot product 0.
        # Float32 arithmetic can still give them a tiny positive similarity.
        model = self.train([
            ["2025-01-01", user, "rating", movie, rating]
            for user, space_rating, romance_rating in (
                ("1", "10", "7"), ("2", "7", "4"), ("3", "7", "2.5")
            )
            for movie, rating in (("space", space_rating), ("romance", romance_rating))
        ])
        self.assertEqual(model["item_neighbors"].nnz, 0)

    def test_likes_and_dislikes_change_cold_start_rankings(self):
        model = self.train([["2025-01-01", "new", "account_created", "", ""]])
        likes = {"likes": "science fiction astronaut spaceship", "dislikes": "romance love"}
        dislikes = {"likes": likes["dislikes"], "dislikes": likes["likes"]}
        preferred = recommend(model, "new", profile=likes)
        reversed_preferences = recommend(model, "new", profile=dislikes)
        self.assertIn(preferred[0]["movie_id"], {"space", "alien"})
        self.assertIn(reversed_preferences[0]["movie_id"], {"romance", "wedding"})
        self.assertLess(preferred[-1]["score"], 0)
        self.assertNotIn("new", model["histories"])

    def test_no_signal_uses_unique_viewers_then_movie_id(self):
        model = self.train([
            ["2025-01-01", "1", "watch", "space", ""],
            ["2025-01-02", "1", "watch", "space", ""],
            ["2025-01-01", "1", "watch", "romance", ""],
            ["2025-01-01", "2", "watch", "romance", ""],
        ])
        expected = ["romance", "space", "alien", "wedding"]
        profiles = (
            None,
            {"likes": "unrecognizedword", "dislikes": ""},
            {"likes": "science fiction", "dislikes": "science fiction"},
        )
        for profile in profiles:
            rows = recommend(model, "unknown", top_k=10, profile=profile)
            self.assertEqual([row["movie_id"] for row in rows], expected)
            self.assertTrue(all(row["score"] == 0 for row in rows))
        self.assertEqual(recommend(model, "unknown"), recommend(model, "unknown"))

    def test_invalid_ratings_and_nonpositive_limits_are_rejected(self):
        for rating in ("0", "11", "nan", "inf", "bad"):
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                self.train([["2025-01-01", "1", "rating", "space", rating]])
        model = self.train([])
        for limit in (0, -1):
            with self.assertRaisesRegex(ValueError, "top_k"):
                recommend(model, "1", top_k=limit)
        for limit in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "n_neighbors"):
                self.train([], n_neighbors=limit)

    def test_compressed_and_plain_csv_read_the_same(self):
        plain = self.data_dir / "movies.csv"
        compressed = self.data_dir / "movies.csv.gz"
        with gzip.open(compressed, "wb") as output:
            output.write(plain.read_bytes())
        self.assertEqual(read_csv(plain), read_csv(compressed))
        model = self.train([])
        self.assertEqual(len(model["movies"]), len(self.movies))


if __name__ == "__main__":
    unittest.main()
