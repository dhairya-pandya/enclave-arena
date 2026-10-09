# Papersseum

A real-time, multi-agent territory-capture tournament. Write an agent, submit it, and watch it fight four others for the map.

Five agents drop into a bounded circular arena for 180 seconds. Leave your land and you draw a trail behind you. Close the loop and everything you enclosed becomes yours. Cut a rival's trail and you don't just kill them, you take their whole territory and get a short speed burst to go after the next one. Whoever owns the most of the map when the clock runs out wins.

Anyone can enter. Your agent can be a few lines of rules, a trained neural net, a search routine, or anything in between. The server runs ladder rounds of random 5-agent lobbies, rates agents by how they place, and a final round sets the leaderboard. Every match is deterministic from its seed and action log, so any game can be replayed and any dispute settled by running it again.

The name is paper plus colosseum: an arena where paper-thin territories are fought over to the last tick.

## The game

- **Arena.** Circular, 11,304 playable cells on a 120x120 grid. The wall is solid. Steer into it and you slide along the edge instead of dying.
- **Match.** Five players, 180 seconds. The engine runs at 30 ticks per second; agents decide 10 times per second.
- **Movement.** Three actions: `0` straight, `1` left, `2` right. You move faster the more land you hold (+1 cell per second for every 5% of the map), and you get a 1.5x burst for 5 seconds after a kill. Speed is capped so you never move more than one cell per tick.
- **Capture.** Re-enter your own land with an open trail and everything the loop encloses becomes yours, including other players' land.
- **Death.** Cut into your own trail and your land goes neutral. Get cut by a rival, or lose a head-on, and the killer takes your land. Either way you respawn with a small fresh patch on empty ground.
- **Score.** Your share of the map at the end. Ties break on time-averaged coverage, then fewer deaths.

A few rules (taking territory on a kill, the kill speed burst, solid walls) are intentional changes from a stock paper.io-style game.

## What your agent sees

Every decision, your agent gets a view built from its own point of view. There are 16 channels: your territory, trail, and head, the same three for each of the four opponents, and the arena mask.

- **Global view**, shape `(16, 120, 120)`. The whole arena, north up. Use it for planning.
- **Local view**, shape `(16, 31, 31)`. Cropped on your head and rotated so your heading points up, so "straight", "left", and "right" always mean the same pixels. Use it for quick reactions.
- **Scalars**: your coverage, rank, speed, boost remaining, time left, and each opponent's coverage, speed, and alive flag.

## Writing an agent

Your file defines a class named `Agent` with `reset(config)` and `act(obs) -> int`:

```python
class Agent:
    def reset(self, config: dict) -> None:
        """Called once per match with the constants and a seed."""

    def act(self, obs: dict) -> int:
        """Called every decision. Return 0, 1 or 2."""
```

Run `papersseum new my_agent.py` to drop a copy-ready starter next to you. Four baselines ship in [`papersseum/agents/`](papersseum/agents): `random`, `greedy` (grinds small capture loops), `safe_expander`, and `hunter` (chases and cuts exposed trails).

If your agent uses trained weights, load them from your own folder via `config["weights_dir"]` in `reset`, with `papersseum.load_weights(config, "policy.npy")` (`.npy`, `.npz`, `.pt`, `.pth`; raw `numpy.load` / `torch.load` are blocked by the scan):

```python
from papersseum import load_weights

def reset(self, config):
    self.policy = load_weights(config, "policy.npy")
```

## Running it

```bash
pip install papersseum             # or: pip install -e ".[dev,viewer]" from a clone

papersseum new my_agent.py                         # write a starter agent
papersseum play my_agent.py --vs greedy,hunter --seed 7 --render match.mp4
papersseum eval my_agent.py --games 30             # benchmark vs a baseline field
papersseum validate my_agent.py                    # the scan + smoke match the server runs on submit
papersseum version                                 # library + engine hash (must match the server)
```

MP4 rendering needs `ffmpeg` on PATH; everything else is pure Python plus NumPy. Or from Python:

```python
import papersseum
result = papersseum.play("my_agent.py", vs=["greedy", "hunter"], seed=7, render="match.mp4")
print(result["placements"])
```

`match.mp4` is a real H.264 video with one frame per engine tick, so 1x playback matches the real 180 second match. `papersseum.ENGINE_HASH` identifies the engine build; a match with the server's hash means local results equal ladder results.

## Limits

A submission is one `.py` file of at most 1 MB (weights live beside it). `reset()` gets 2 seconds and each `act()` gets 50 ms; a late or invalid answer counts as action `0` (straight). Only a curated set of imports is allowed (numpy, math, collections, and similar; `torch` for weights), and `open`, `eval`, `getattr`, `vars` and raw `numpy.load` / `torch.load` are blocked. `papersseum validate` runs the same scan locally.

## Replays

Every match is a JSONL file (add `.gz` to compress): a header line (seed, engine hash, numpy version, players), one line of five actions per decision, and a result line. A match is about 1 KB gzipped.

```bash
papersseum play my_agent.py --save-replay game.jsonl.gz
papersseum render game.jsonl.gz game.mp4
```

```python
from papersseum.replay_io import load_replay
from papersseum import replay_match, iter_frames

rep = load_replay("game.jsonl.gz")
same = replay_match(rep["seed"], rep["action_log"])          # re-run without observations
for frame in iter_frames(rep["seed"], rep["action_log"]):    # owner, trail, heads per decision
    ...
```

## Test and iterate locally

Measure your agent against a field of baselines before you ever submit:

```bash
papersseum eval examples/my_agent.py --games 30     # placements, win rate, coverage, rating, strength
```

It plays your agent in slot 0 against a varied baseline field over many seeds and reports the same signals the ladder uses: how often you place 1st to 5th, mean coverage, and a rating (online Elo plus a batch Plackett-Luce strength).

See exactly what your agent sees, as one character per cell (`@` your head, `M` your land, `t` your trail, `O` opponent land, `~` opponent trail, `X` opponent head, `.` empty, `#` wall):

```python
import papersseum
from papersseum import channels as ch

env = papersseum.PapersseumEnv(seed=7)
obs = env.reset()
print(papersseum.ascii_obs(obs[0], "local"))         # your agent's eyes
on_own_land = obs[0]["local"][ch.OWN_TERRITORY, 15, 15] > 0.5
```

Use `papersseum.channels` for named indices (`OWN_TERRITORY`, `OWN_TRAIL`, `OWN_HEAD`, `opponent(i)`, `ARENA_MASK`) instead of magic numbers.

## Status

The engine, reference environment, baselines, self-eval, JSONL replays and rating are done. The tournament platform (Docker-sandboxed match workers, queue, ladder and web app) is built separately and uses this package as its engine.

## Running matches in a sandbox (platform operators)

```python
from papersseum.sandbox import run_sandboxed_match, validate_sandboxed, DockerBackend

backend = DockerBackend(image="papersseum-sandbox")     # build with docker/Dockerfile
check = validate_sandboxed("agent.py", backend=backend)  # {ok, error, violations, warnings, stats}
result = run_sandboxed_match(seed, [path0, path1, path2, path3, path4], backend=backend)
open("match.jsonl", "wb").write(result["replay"])
```

Each agent gets its own container (no network, no capabilities, read-only root, 512 MB, 1 CPU, 64 processes, code mounted read-only, no environment). Agents think in parallel; a late `act()` becomes action 0 plus a strike, and an agent that exits or goes silent is killed and plays straight for the rest of the match. Time is measured by the runner, not the agent. An agent late on more than 20% of decisions (or keeping the match waiting more than 20 s in total) is disqualified as `too_slow`; occasional slow decisions only cost that move. `validate_sandboxed` plays a full match (the agent against itself plus three baselines) under the same rule. `SubprocessBackend` (the default) is for tests and development only and is not a security boundary.
