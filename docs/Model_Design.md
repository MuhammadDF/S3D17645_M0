# How the model works and why I changed it

I switched users with history to item-based collaborative filtering. I kept the
LLM and content-based approach for new users. The main idea is:

**Your ratings tell the model what you like. Other people's ratings help it
figure out which movies are connected.** Unrated watches also count, but as weaker
evidence of interest.

## What changed

| Approach | How it finds a connection |
| --- | --- |
| Previous content-based model | These movies share words in their titles, genres, or plots. |
| Current collaborative model | These movies get similar reactions from the people who watched or rated them. |

The previous model combined descriptions of liked movies and subtracted those of
disliked movies. TF-IDF gives distinctive words more weight than common ones.

Collaborative filtering uses audience reactions. A comedy and a thriller can be
connected despite different descriptions. One person's high rating is not enough
to recommend a movie to everyone.

## What training actually does

Training builds a table of movies and users. Each recorded interaction becomes
a preference number:

- A rating becomes `(rating - 5.5) / 4.5`: 10 becomes +1, 1 becomes -1.
- An unrated watch gets +0.25. Repeated watch events do not add more weight.
- The latest rating replaces the watch weight. Missing interactions mean unknown,
  not disliked. Sparse storage avoids storing all those empty entries.

The model compares movies using cosine similarity, a measure of how closely their
preference patterns line up. Shared likes and shared dislikes both contribute to
a connection; opposite reactions weaken it. At least two users must have watched
or rated both movies. Only positive connections are kept, ignoring tiny rounding
errors and a movie's connection to itself.

Training saves up to 20 strongest connections per candidate movie. These are its
*neighbors*. KNN means *k-nearest neighbors*; `k = 20` is this limit. Separately,
`--top-k` controls how many recommendations come back. This preparation is the
learning step, without a neural network.

## A few made-up scenarios

### 1. You like a comedy and get a thriller

You give Comedy A a 9/10. Several people who liked Comedy A also liked Thriller B,
so the model learns a connection between them.

If A is among B's saved neighbors, your high rating helps B's score. B can be
recommended despite the different genre because of that shared taste.

### 2. You dislike something that other people like

You give Comedy A a 1/10. That lowers the score of connected movies, including B.
Other people's ratings establish the connection; they do not replace your rating.

If B also connects to movies you liked, the model combines that evidence using a
similarity-weighted average. For example, a connection of 0.8 to a movie you rated
10 and 0.2 to one you rated 1 gives:

```text
(0.8 × 1 + 0.2 × -1) / (0.8 + 0.2) = 0.6
```

This score orders recommendations. It is not a predicted rating or a probability.

### 3. Your description says “I hate thrillers”

If you have history, the model does not read that description or check movie
content. Thriller B can still be recommended through its connection to Comedy A.
Your low ratings matter, but written genre dislikes are not restrictions on this path.

### 4. You are new and say “I like comedies and hate horror”

With no history, the model uses the other path. The existing `qwen3:8b` LLM extracts
likes and dislikes. TF-IDF then matches those preferences against titles, genres,
and plots. Comedy matches raise scores; horror matches lower them. Dislikes are
penalties, not absolute bans. The LLM does not choose the final movie IDs.

Training learns the vocabulary and word weights from the movie catalog. Genres
are repeated three times to give them more influence. The content model subtracts
disliked features from liked features, then compares the result with each movie
using cosine similarity.

### 5. There is very little history

Even one watch selects the collaborative path. A candidate with no matching
neighbors gets zero, which can rank above a negative score. Ties use popularity,
measured by distinct users who watched or rated a movie, then movie ID.

With only one matching neighbor, the average equals your preference for that
neighbor, even if the connection is weak. A high score does not mean strong evidence.
Without history or usable content preferences, popularity decides the ranking.
Already watched or rated movies are always excluded.

## What this gains and loses

The change can uncover connections that movie descriptions miss, even when those
descriptions are incomplete. It also depends on enough shared history. A movie
with no interactions cannot get a personalized collaborative score. The median
movie here has two ratings, and 1,175 of 2,569 movies have no qualifying neighbors.
Two shared users can still be weak
evidence. Popular movies can dominate when personalized scores are missing.

**The model chooses one path per user. It does not blend ratings and content.**
I have not measured an accuracy improvement. Comparing predictions against
interactions held out from training would be needed to check that.

The [first suggested paper](https://www.nature.com/articles/s41598-025-15096-4)
provides collaborative filtering background. The
[second paper](https://arxiv.org/html/2509.12948v1) discusses a more involved
two-tower approach: one neural network represents users and another represents
movies, helping narrow a large catalog quickly. I chose this small baseline to
keep the implementation understandable. It uses the existing NumPy, SciPy, and
scikit-learn dependencies, with no new packages or serving infrastructure.

## Why I chose a local LLM

I chose the relatively small `qwen3:8b` model to make this setup easier for other
students to reproduce. The aim was a manageable local setup without a paid API
or cloud server. Its job is to read short likes and dislikes, rather than choose
all the recommendations. I did not need the most capable LLM for that starting
point, and the saved outputs show where its extractions still make mistakes.

Ollama is the program that runs this model locally. It is a prerequisite for
generating those outputs, and is installed separately from the Python packages.
The saved JSON lets others repeat the recommendations without running the LLM.
Regenerating text can still produce different wording. The
[README](../README.md#ollama-prerequisite-and-saved-llm-results) gives the setup
commands and hardware considerations; a smaller model still needs disk space
and memory.
