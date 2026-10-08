"""Evil-agent suite: every hostile or broken agent gets a defined outcome.

Runs on SubprocessBackend (no isolation, so only behaviours the runner itself
enforces). Isolation cases (memory, fork, network) need Docker and are skipped
without it.
"""

import os
import shutil

import pytest

import papersseum as p
from papersseum.agents import RandomAgent
from papersseum.sandbox import run_sandboxed_match, validate_sandboxed, DockerBackend

HEAD = "from papersseum import Agent\n"


def play(tmp_path, src, n=15, **kw):
    f = tmp_path / "a.py"
    f.write_text(src)
    kw.setdefault("hard_timeout_s", 0.5)
    lobby = [str(f)] + [RandomAgent() for _ in range(4)]
    return run_sandboxed_match(3, lobby, max_decisions=n, **kw)


def test_good_agent_matches_in_process_run(tmp_path):
    src = open("examples/my_agent.py").read()
    f = tmp_path / "a.py"
    f.write_text(src)
    res = run_sandboxed_match(7, [str(f)] + [p.BASELINES["greedy"]() for _ in range(4)], max_decisions=60)
    from papersseum.loader import load_agent_from_file
    ref = p.run_match(7, [load_agent_from_file(str(f))()] + [p.BASELINES["greedy"]() for _ in range(4)])
    assert res["slots"][0]["status"] == "ok"
    assert res["action_log"] == ref["action_log"][:60]
    assert res["replay"].startswith(b'{"type":"header"')


def test_infinite_loop_is_killed_and_plays_straight(tmp_path):
    res = play(tmp_path, HEAD + "class Agent(Agent):\n    def act(self, obs):\n        while True: pass\n")
    s = res["slots"][0]
    assert s["status"] == "unresponsive"
    assert len(res["action_log"]) == 15                      # match still finished
    assert all(row[0] == 0 for row in res["action_log"])


def test_slow_agent_gets_strikes_not_killed(tmp_path):
    src = HEAD + "import time\nclass Agent(Agent):\n    def act(self, obs):\n        time.sleep(0.08)\n        return 1\n"
    res = play(tmp_path, src, n=8)
    s = res["slots"][0]
    assert s["status"] == "ok" and s["strikes"] == 8
    assert all(row[0] == 0 for row in res["action_log"])     # late answers never applied


def test_exception_and_invalid_return_become_action_zero(tmp_path):
    for body in ("raise RuntimeError('boom')", "return 7", "return True", "return 'left'"):
        res = play(tmp_path, HEAD + f"class Agent(Agent):\n    def act(self, obs):\n        {body}\n", n=5)
        s = res["slots"][0]
        assert s["status"] == "ok" and s["errors"] == 5, body
        assert all(row[0] == 0 for row in res["action_log"])


def test_reset_raising_or_hanging_is_rejected(tmp_path):
    res = play(tmp_path, HEAD + "class Agent(Agent):\n    def reset(self, c):\n        raise ValueError('no')\n", n=3)
    assert res["slots"][0]["status"] == "reset_failed" and "ValueError" in res["slots"][0]["error"]
    res = play(tmp_path, HEAD + "class Agent(Agent):\n    def reset(self, c):\n        while True: pass\n", n=3,
               reset_timeout_s=1.0)
    assert res["slots"][0]["status"] == "reset_failed"


def test_import_time_hang_and_syntax_error(tmp_path):
    res = play(tmp_path, "while True: pass\n", n=3, reset_timeout_s=1.0)
    assert res["slots"][0]["status"] == "reset_failed"
    res = play(tmp_path, "def (:\n", n=3)
    assert res["slots"][0]["status"] == "reset_failed"


def test_process_exit_is_a_crash(tmp_path):
    src = HEAD + "import os\nclass Agent(Agent):\n    def act(self, obs):\n        os._exit(1)\n"
    res = play(tmp_path, src, n=5)
    assert res["slots"][0]["status"] == "crashed"
    assert len(res["action_log"]) == 5


def test_print_spam_cannot_corrupt_the_protocol(tmp_path):
    src = HEAD + "class Agent(Agent):\n    def act(self, obs):\n        print('x' * 5000)\n        return 1\n"
    res = play(tmp_path, src, n=10)
    assert res["slots"][0]["status"] == "ok" and res["slots"][0]["errors"] == 0


def test_environment_is_not_leaked(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "secret")
    src = HEAD + "import os\nclass Agent(Agent):\n    def act(self, obs):\n        return 1 if 'SUPABASE_SERVICE_KEY' in os.environ else 0\n"
    res = play(tmp_path, src, n=5)
    assert all(row[0] == 0 for row in res["action_log"])


def test_validate_sandboxed_outcomes(tmp_path):
    def check(src):
        f = tmp_path / "v.py"
        f.write_text(src)
        return validate_sandboxed(str(f), decisions=10)
    assert check(open("examples/my_agent.py").read())["ok"]
    r = check("import os\n" + HEAD + "class Agent(Agent): pass\n")
    assert not r["ok"] and r["violations"] and "os" in r["error"]
    assert not check(HEAD + "class Agent(Agent):\n    def act(self, obs):\n        raise KeyError(1)\n")["ok"]
    assert not check("x = 1\n")["ok"]                          # no Agent class
    assert not check("a = '" + "x" * 1_100_000 + "'\n")["ok"]  # over 1 MB


def test_wrong_lobby_size_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_sandboxed_match(1, [RandomAgent()])


def test_docker_command_is_locked_down():
    cmd = DockerBackend(image="img").command("n", "/tmp/x")
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--memory 512m",
                 "--pids-limit 64", "no-new-privileges", "/tmp/x:/agent:ro", "--user 65534:65534"):
        assert flag in joined
    assert "-e " not in joined and "--env" not in joined       # no environment passed in


needs_docker = pytest.mark.skipif(shutil.which("docker") is None or not os.environ.get("PAPERSSEUM_DOCKER_TESTS"),
                                  reason="set PAPERSSEUM_DOCKER_TESTS=1 with docker + papersseum-sandbox image")


@needs_docker
@pytest.mark.parametrize("name,body", [
    ("memory_bomb", "x = bytearray(2_000_000_000)\n"),
    ("fork_bomb", "import os\nwhile True: os.fork()\n"),
    ("network", "import socket\ns = socket.create_connection(('1.1.1.1', 80), 2)\n"),
    ("write_fs", "open('/agent/x', 'w').write('x')\n"),
])
def test_docker_isolation(tmp_path, name, body):
    src = HEAD + body + "class Agent(Agent):\n    def reset(self, c):\n        pass\n"
    res = play(tmp_path, src, n=3, backend=DockerBackend(), reset_timeout_s=5)
    assert res["slots"][0]["status"] == "reset_failed"


def test_sustained_slowness_disqualifies_and_match_still_finishes(tmp_path):
    src = HEAD + "import time\nclass Agent(Agent):\n    def act(self, obs):\n        time.sleep(0.3)\n        return 1\n"
    res = play(tmp_path, src, n=120, hard_timeout_s=1.0)
    s = res["slots"][0]
    assert s["status"] == "too_slow"
    assert len(res["action_log"]) == 120                     # nobody kept waiting
    assert all(row[0] == 0 for row in res["action_log"])


def test_occasional_slowness_is_tolerated(tmp_path):
    src = (HEAD + "import time\nclass Agent(Agent):\n    n = 0\n    def act(self, obs):\n"
           "        self.n += 1\n        if self.n % 20 == 0:\n            time.sleep(0.12)\n        return 1\n")
    res = play(tmp_path, src, n=100)
    s = res["slots"][0]
    assert s["status"] == "ok" and 0 < s["strikes"] <= 6      # 5 late decisions, each played straight


def test_agent_cannot_hide_slowness_by_patching_the_clock(tmp_path):
    src = (HEAD + "import time\nclass Agent(Agent):\n    def reset(self, c):\n"
           "        time.perf_counter = lambda: 0.0\n    def act(self, obs):\n"
           "        time.sleep(0.2)\n        return 1\n")
    res = play(tmp_path, src, n=10)
    assert res["slots"][0]["strikes"] == 10                    # reported 0 ms, runner's clock disagreed
    assert all(row[0] == 0 for row in res["action_log"])


def test_validation_rejects_sustained_slowness(tmp_path):
    f = tmp_path / "slow.py"
    f.write_text(HEAD + "import time\nclass Agent(Agent):\n    def act(self, obs):\n        time.sleep(0.12)\n        return 1\n")
    r = validate_sandboxed(str(f), decisions=80)
    assert not r["ok"] and "too slow" in r["error"]
