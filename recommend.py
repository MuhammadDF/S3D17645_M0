"""Load the trained model and recommend unseen movies for one user."""

import argparse
import pickle
from pathlib import Path

from cold_start import cached_preferences, descriptions, load_cache
from model import read_csv, recommend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pkl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--profiles", type=Path, default=Path("data/cold_start_profiles.json"))
    parser.add_argument("--show-titles", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if not args.model.exists():
        parser.error("No trained model found. Run `python3 train.py` first.")

    # Only load model files you trained yourself: pickle can execute Python code.
    with args.model.open("rb") as stream:
        model = pickle.load(stream)
    if model.get("model_type") != "item_cf_with_content_cold_start":
        parser.error("This model uses the previous design. Run `python3 train.py` again.")

    profile = None
    method = "item-based collaborative filtering from viewing and rating history"
    if args.user_id not in model["histories"]:
        method = "popularity fallback"
        users = read_csv(args.data_dir / "users.csv.gz")
        user = next((row for row in users if row["user_id"] == args.user_id), None)
        if user and any(descriptions(user).values()):
            profile = cached_preferences(load_cache(args.profiles), user)
            if profile is None:
                parser.error("Missing or outdated LLM preferences for this user. Run "
                             f"`python3 cold_start.py --user-id {args.user_id}` first "
                             "(use matching --data-dir and --output for custom paths).")
            method = "content matching with LLM preferences from the user's description"

    results = recommend(model, args.user_id, args.top_k, profile)
    if args.show_titles:
        print(f"User {args.user_id}: {method}")
        if profile:
            print(f"Likes: {profile['likes']}")
            print(f"Dislikes: {profile['dislikes']}")
        for rank, movie in enumerate(results, start=1):
            print(f"{rank:2}. {movie['title']} ({movie['genres']})\n"
                  f"    {movie['movie_id']} | score: {movie['score']:.3f}")
    else:
        print(",".join(movie["movie_id"] for movie in results))


if __name__ == "__main__":
    main()
