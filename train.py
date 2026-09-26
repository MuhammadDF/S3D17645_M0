"""Train item-based collaborative filtering and the content model for new users."""

import argparse
import pickle
import time
from pathlib import Path

from model import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pkl"))
    args = parser.parse_args()

    start = time.perf_counter()
    model = train_model(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(model, stream)
    print(f"Trained on {len(model['movies']):,} movies and "
          f"{len(model['histories']):,} users with history.")
    print(f"Saved {args.output} in {time.perf_counter() - start:.2f} seconds.")


if __name__ == "__main__":
    main()
