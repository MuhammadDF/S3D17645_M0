# Movie recommendations: Milestone 0

A user's own ratings tell the model what they like. Other people's ratings help
it figure out which movies are connected. This is called **collaborative filtering**.

For someone with no history, there is nothing to compare yet. A language model
(LLM) reads their written likes and dislikes, and the recommender matches those
preferences to movie titles, genres, and descriptions instead.

These are two separate approaches. **The model does not combine ratings and
movie content for the same user.** The [design notes](docs/Model_Design.md) walk
through examples and explain why I made this choice. The submission report is
[m0_report.md](m0_report.md).

## Running the model

Run these commands from the repository root. The three data files and saved LLM
results are already in `data/`.

Python and the packages below are required for training and recommendations.
**Ollama and `qwen3:8b` are also prerequisites for generating LLM preferences.**
They are installed separately from `requirements.txt`; the setup is explained
[below](#ollama-prerequisite-and-saved-llm-results). The supplied examples can
reuse the saved preferences without Ollama installed or running.

I tested with Python 3.13.5 and the installed NumPy, SciPy, and scikit-learn
versions in `requirements.txt`. I did not add dependencies for the model change.
On a fresh machine, set up the same environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Then train and get recommendations:

```bash
python3 train.py
python3 recommend.py 1 --top-k 5 --show-titles
python3 recommend.py 1001 --top-k 5 --show-titles
```

Training is the preparation step. It reads the data, learns movie connections
and word weights, and saves the result in `artifacts/model.pkl`. The recommendation
command loads that file, so training does not need to run for every request.

User `1` has history and gets recommendations from rating patterns. User `1001`
is new and gets recommendations from their saved written preferences.
`--show-titles` prints names, scores, and which approach was used. Without it,
the output is comma-separated movie IDs. `--top-k 5` asks for five movies;
the default is 20. The scores help order movies. They are not predicted star ratings.

Retrain after changing the model. The command will ask for this if it finds a
file from the previous design. I keep the generated model out of Git because
it can be rebuilt. Only load trusted pickle files, since pickle can execute code.

## What happens when someone asks for recommendations?

| What is known about the user? | What happens? |
| --- | --- |
| They have watched or rated something | Their history is matched to movies connected through other users' ratings. |
| They have no history but wrote likes and dislikes | The LLM reads that description, then movie content is matched to the extracted preferences. |
| There is no usable information | The model returns popular movies. |

For a made-up example, suppose a user gave **Comedy A a 9/10**. Other people
who liked A also tended to like **Thriller B**, so the model saved a connection
between them. If A is one of B's saved connections, liking A raises B's score
for that user. Giving A a 1/10 would lower it instead.

That is why someone can get a different genre from the one they usually watch.
The connection comes from people's reactions, not from matching genres. For a
user with history, the model does not read their written likes and dislikes,
even if the description says they hate thrillers. Even one watch takes this route.

A new user who writes "I like comedies and dislike horror" takes the other route.
The LLM separates those preferences. The content model gives matching comedy
features a positive contribution and horror features a negative one. Dislikes
lower scores; they do not completely ban a genre.

Training keeps up to 20 related movies, called **neighbors**, for each movie.
A connection requires at least two people who watched or rated both movies.
This limit of 20 is separate from how many results `--top-k` asks for.

If a movie has no useful connection to the user's history, its score is zero.
Ties are broken by popularity, then movie ID. Popularity here means the number
of different people who watched or rated the movie, not its average star rating.
A movie with no evidence can rank above one with negative evidence. Watched or
rated movies are always removed from the results.

## Ollama prerequisite and saved LLM results

Ollama is the program that runs the LLM on the computer. `qwen3:8b` is the model
it runs. Both are needed to generate preferences for new descriptions or to
regenerate the saved results.

I chose a relatively small local 8B model to make the setup easier for other
students to reproduce on their own computers, without a paid API or cloud server.
Its job is limited to reading short movie preferences. I prioritized a manageable
setup over finding the most capable LLM. The model is already trained; this
project only asks it to extract preferences.

The [model download](https://ollama.com/library/qwen3:8b) is about 5.2 GB.
Running it also needs available memory, so this choice does not guarantee good
performance on every computer. [Ollama's memory notes](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests)
explain why runtime settings affect memory use.

I used the model that was already installed locally and saved the results
for all 50 new users in [cold_start_profiles.json](data/cold_start_profiles.json).
The file includes the original descriptions, extracted preferences, prompt,
and model name. Training reads movies and events, without calling the LLM.
New-user recommendations read the saved preferences. Neither step needs Ollama
running when the required results are already saved, and no API key is needed.

To reproduce the LLM step on a fresh machine, first
[install Ollama](https://ollama.com/download). Open the app, or run
`ollama serve` in a separate terminal if it is not already running. Then download
the model once and check that it appears in the list:

```bash
ollama pull qwen3:8b
ollama list
```

With Ollama running, generate any missing results:

```bash
python3 cold_start.py
```

These commands follow [Ollama's CLI instructions](https://docs.ollama.com/cli).
The script talks to Ollama on `localhost:11434` using Python's standard library.
It saves after each user and skips results that are still current, so an
interrupted run can resume.
To make a fresh LLM call for just one user:

```bash
python3 cold_start.py --user-id 1002 --refresh
```

User `1002` wrote titles such as "fite club" and "usuall suspects". The prompt
asks the LLM to correct spelling and keep likes separate from dislikes. That is
the LLM's contribution; the recommender still chooses the final movies.

Without `--user-id`, `--refresh` regenerates all 50 results. New LLM calls can
produce different wording across machines or model versions. I keep the saved
JSON in Git so teammates can use the same inputs. If a description or the prompt
changes, the old result needs to be generated again.

If a described new user's saved result is missing or outdated, the recommendation
command stops and explains how to generate it. It does not silently replace that
required LLM step with popularity. Invalid LLM responses are rejected before saving.

## Reading and checking the code

| File | What it does |
| --- | --- |
| [model.py](model.py) | Reads events, learns movie connections and text features, and picks recommendations. |
| [train.py](train.py) | Runs the preparation step and saves the model. |
| [cold_start.py](cold_start.py) | Asks the LLM to extract new users' likes and dislikes and saves them. |
| [recommend.py](recommend.py) | Loads the model and returns recommendations for a user ID. |

Run the checks with:

```bash
python3 -m unittest discover -s tests -v
```

All 17 tests pass. I also checked all 1,050 supplied users: each gets 20 valid,
unique movies that are not in their history. All 50 new users keep the same
results as before the switch to collaborative filtering.

These checks show that the code follows its rules. They do not show that people
will like the recommendations. I have not compared accuracy using ratings left
out of training, so I cannot say this version recommends better movies.

Some movies have too few ratings to form useful connections. The LLM can also
misread a description, such as turning a dislike of one movie into a dislike of
its whole genre. The design notes explain these limits with examples. The model
does not use demographics, license costs, or external datasets.

## My milestone 0 submission

My deliverable is the repository, including the code, three datasets, saved LLM
results, dependency list, instructions, tests, and root `m0_report.md`.
The Markdown report meets the requested format; a PDF is optional.
The implementation, LLM step, report, and run instructions are complete for
milestone 0.
