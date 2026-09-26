# Milestone 0: Individual Recommendation Model

## Learning

I used `events.csv.gz`, `movies.csv.gz`, and `users.csv.gz`: 2,569 movies,
1,050 users, 27,327 watches, 27,327 ratings, and 50 account creation events.
The 50 new users have written preferences but no history. The model does not use
demographics, license costs, or external data.

For users with history, their own ratings tell the model what they like. Ratings
across users help it find connections between movies. This is called
**item-based collaborative filtering**.

For example, suppose several people react similarly to Comedy A and Thriller B.
The model can connect those movies. Someone who liked A could then get B, even
though the genres differ. Someone who disliked A would lower B's score. One
other person's high rating does not guarantee a recommendation.

I switched from matching movie descriptions because shared tastes can connect
movies with different wording or genres. I chose **K-nearest neighbors (KNN)**:
keep a short list of related movies for each movie. The code uses the existing
NumPy, SciPy, and scikit-learn dependencies.

The [training code](model.py) turns ratings into preference weights using
`(rating - 5.5) / 4.5`: 10 becomes +1 and 1 becomes -1. An unrated watch gets
+0.25. Repeated watches count once; the latest rating takes priority. Missing
ratings do not count as dislikes. Training compares these patterns using cosine
similarity and stores up to 20 similar movies per movie. Each connection needs
at least two shared users. Every watched movie in the supplied data has a rating.

For each unseen movie, the [recommendation code](model.py) combines the user's
ratings for its connected movies. Stronger connections get more weight. High
ratings help and low ratings penalize the score. Without a useful connection,
the score is zero. Popularity breaks ties. Watched or rated movies are excluded.

Even one watch uses this route. It ignores movie content and written preferences,
so it can recommend a genre the user dislikes. Sparse history also limits the
connections available. The [design notes](docs/Model_Design.md) explain these
tradeoffs.

For new users, the [cold-start code](cold_start.py) asks a local `qwen3:8b` LLM
through Ollama to organize their written likes and dislikes into titles, genres,
and themes, correcting obvious misspellings. I chose this smaller local 8B model
to make the setup easier for other students to reproduce without a paid API.
I prioritized a manageable setup over the most capable LLM. I saved all 50 outputs
in [cold_start_profiles.json](data/cold_start_profiles.json), so normal runs do not
need Ollama.

The content model then matches those preferences against movie titles, genres,
and summaries. Training learns its vocabulary and word weights from the movie
catalog. This weighted word matching is called **TF-IDF**, where common words
matter less. Genres are repeated three times to give them more influence.
Cosine similarity compares movies with
the liked features minus the disliked features. Dislikes lower scores without
banning movies. Unusable preferences fall back to popularity. The two routes
do not blend ratings and content for the same user.

I checked the implementation, but have not measured recommendation accuracy or
established that this version is better than the original.

## Running the model

I tested with Python 3.13.5 and the installed versions in
[`requirements.txt`](requirements.txt). On a fresh machine, create a virtual
environment and run `python3 -m pip install -r requirements.txt`.
From the repository root, run:

```bash
python3 train.py
python3 recommend.py 1 --top-k 5 --show-titles
python3 recommend.py 1001 --top-k 5 --show-titles
```

Training saves `artifacts/model.pkl`. User `1` uses collaborative filtering;
user `1001` uses saved LLM preferences and content matching. Without
`--show-titles`, the output is comma-separated movie IDs; the default is 20 movies.
Ollama and `qwen3:8b` are prerequisites for generating LLM results and are
installed separately from the Python packages. With Ollama running, use
`python3 cold_start.py --refresh`. Saved results need no running LLM.
Run tests with `python3 -m unittest discover -s tests -v`.
The [README](README.md) gives the full setup instructions.
