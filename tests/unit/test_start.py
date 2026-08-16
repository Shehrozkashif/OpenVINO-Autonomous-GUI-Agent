# tests/unit/test_start.py
"""Unit tests for start.py — the setup and server-launch script.

Only the parts that decide whether setup can proceed are covered here: nothing
downloads, converts, or launches a process. The failure these pin was real. A
model folder was missing generation_config.json, so OVMS refused the servable
six seconds in, but setup reported the model "already in repository" (config.json
still listed it) and then watched a progress counter for the full ten-minute
timeout before printing a message that named no cause.
"""
import json
import os

import pytest

import start


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """An OVMS model repository, redirected away from the real one."""
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(start, "_REPO", str(models))
    monkeypatch.setattr(start, "_CONFIG_JSON", str(models / "config.json"))
    return models


def _register(repo, *names):
    (repo / "config.json").write_text(json.dumps(
        {"mediapipe_config_list": [{"name": n} for n in names]}))


def _write_ir(repo, name):
    d = repo / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "openvino_model.xml").write_text("<net/>")


# ── Is the model really there? ────────────────────────────────────────────────

class TestModelAlreadyExported:

    def test_registered_and_on_disk(self, repo):
        _register(repo, "qwen3-8b-int4-ov")
        _write_ir(repo, "qwen3-8b-int4-ov")
        assert start._model_already_exported("qwen3-8b-int4-ov")

    def test_registered_but_folder_missing(self, repo):
        """config.json outlives the weights — a moved or half-copied repo."""
        _register(repo, "qwen3-8b-int4-ov")
        assert not start._model_already_exported("qwen3-8b-int4-ov")

    def test_registered_but_no_ir_in_folder(self, repo):
        """The folder exists and holds files, but the export never wrote IR."""
        (repo / "qwen3-8b-int4-ov").mkdir()
        (repo / "qwen3-8b-int4-ov" / "config.json").write_text("{}")
        _register(repo, "qwen3-8b-int4-ov")
        assert not start._model_already_exported("qwen3-8b-int4-ov")

    def test_on_disk_but_not_registered(self, repo):
        _write_ir(repo, "qwen3-8b-int4-ov")
        _register(repo)
        assert not start._model_already_exported("qwen3-8b-int4-ov")

    def test_no_config_json_at_all(self, repo):
        assert not start._model_already_exported("qwen3-8b-int4-ov")

    def test_vlm_ir_is_named_differently(self, repo):
        """A VLM writes openvino_language_model.xml, not openvino_model.xml."""
        d = repo / "ui-tars-1.5-7b-int8-ov"
        d.mkdir()
        (d / "openvino_language_model.xml").write_text("<net/>")
        _register(repo, "ui-tars-1.5-7b-int8-ov")
        assert start._model_already_exported("ui-tars-1.5-7b-int8-ov")


# ── Did OVMS give up? ─────────────────────────────────────────────────────────

# Verbatim from a real ovms.log: the VLM folder was missing generation_config.json.
FAILED_LOG = """\
[2026-08-15 12:32:30.710][4764][modelmanager][info][pipelinedefinitionstatus.hpp:60] Mediapipe: qwen3-8b-int4-ov state changed to: AVAILABLE after handling: ValidationPassedEvent:
[2026-08-15 12:32:30.711][4764][modelmanager][info][servable_initializer.cpp:450] Initializing Visual Language Model Continuous Batching servable
[2026-08-15 12:32:30.711][4764][modelmanager][info][pipelinedefinitionstatus.hpp:60] Mediapipe: ui-tars-1.5-7b-int8-ov state changed to: LOADING_PRECONDITION_FAILED after handling: ValidationFailedEvent:
[2026-08-15 12:32:30.711][4764][serving][error][modelmanager.cpp:573] Failed to process mediapipe graph config:Check 'f.is_open()' failed at src\\cpp\\src\\generation_config.cpp:25:
Failed to open 'models\\ui-tars-1.5-7b-int8-ov\\./generation_config.json' with generation config
"""

HEALTHY_LOG = """\
[2026-08-15 12:32:24.784][4764][serving][info][drogon_http_server.cpp:187] REST server listening on port 8000
[2026-08-15 12:32:24.787][4764][modelmanager][info][servable_initializer.cpp:439] Initializing Language Model Continuous Batching servable
[2026-08-15 12:32:30.710][4764][modelmanager][info][pipelinedefinitionstatus.hpp:60] Mediapipe: qwen3-8b-int4-ov state changed to: AVAILABLE after handling: ValidationPassedEvent:
"""


class TestOvmsLoadFailure:

    def _log(self, tmp_path, text):
        p = tmp_path / "ovms.log"
        p.write_text(text)
        return str(p)

    def test_reports_the_reason_not_just_the_state(self, tmp_path):
        failure = start._ovms_load_failure(self._log(tmp_path, FAILED_LOG))
        assert failure
        # the cause has to survive, not only the state change
        assert "generation_config.json" in failure

    def test_silent_while_the_load_is_healthy(self, tmp_path):
        assert start._ovms_load_failure(self._log(tmp_path, HEALTHY_LOG)) == ""

    def test_silent_before_the_log_exists(self, tmp_path):
        assert start._ovms_load_failure(str(tmp_path / "nope.log")) == ""

    def test_no_log_path(self):
        assert start._ovms_load_failure("") == ""

    def test_partial_log_is_not_a_crash(self, tmp_path):
        """The log is read while OVMS is still writing it."""
        cut = FAILED_LOG[:FAILED_LOG.index("LOADING_PRECONDITION_FAILED") + 10]
        assert start._ovms_load_failure(self._log(tmp_path, cut)) == ""

    @pytest.mark.parametrize("state", ["LOADING_FAILED", "LOADING_PRECONDITION_FAILED"])
    def test_every_terminal_state_is_caught(self, tmp_path, state):
        log = f"[info] Mediapipe: m state changed to: {state} after handling: X\n"
        assert start._ovms_load_failure(self._log(tmp_path, log))


class TestWaitForOvms:

    def test_gives_up_as_soon_as_the_log_says_so(self, tmp_path, monkeypatch):
        """The whole point: seconds, not the full 600 s timeout."""
        log = tmp_path / "ovms.log"
        log.write_text(FAILED_LOG)
        monkeypatch.setattr(start.time, "sleep", lambda s: None)
        monkeypatch.setattr(start, "_both_servables_ready", lambda: False)
        written = []
        monkeypatch.setattr(start.sys.stdout, "write", lambda s: written.append(s))
        assert start._wait_for_ovms(str(log)) is False
        polls = [s for s in written if "Waiting for OVMS" in s]
        assert not polls, "kept counting past a failure it had already seen"

    def test_success_still_wins_over_a_stale_error(self, tmp_path, monkeypatch):
        """A servable that recovered must not be failed by an old log line."""
        log = tmp_path / "ovms.log"
        log.write_text(FAILED_LOG)
        monkeypatch.setattr(start.time, "sleep", lambda s: None)
        monkeypatch.setattr(start, "_both_servables_ready", lambda: True)
        assert start._wait_for_ovms(str(log)) is True


def test_repo_path_is_relative_to_start_py_not_the_shell(monkeypatch):
    """Running from another directory must not silently retarget the repo."""
    assert start._REPO.startswith(os.path.dirname(os.path.abspath(start.__file__)))


# ── generation_config.json ────────────────────────────────────────────────────
# OVMS refuses any servable without this file, and UI-TARS-1.5-7B does not ship
# one upstream — so whether the export produced it came down to what optimum
# could infer. When it did not, setup declared the model ready and OVMS failed
# on a path that does not exist, which reads like a broken download.

# The real values from ByteDance-Seed/UI-TARS-1.5-7B's config.json.
_UITARS_IDS = {"bos_token_id": 151643, "eos_token_id": 151645}


class TestEnsureGenerationConfig:

    def _model(self, repo, name, config: dict | None):
        d = repo / name
        d.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (d / "config.json").write_text(json.dumps(config))
        return d

    def test_builds_it_from_the_models_own_config(self, repo):
        d = self._model(repo, "ui-tars", _UITARS_IDS)
        assert start._ensure_generation_config("ui-tars")
        written = json.loads((d / "generation_config.json").read_text())
        assert written["eos_token_id"] == 151645
        assert written["bos_token_id"] == 151643
        assert written["pad_token_id"] == 151643

    def test_existing_file_is_left_alone(self, repo):
        d = self._model(repo, "ui-tars", _UITARS_IDS)
        (d / "generation_config.json").write_text('{"eos_token_id": 42}')
        assert start._ensure_generation_config("ui-tars")
        assert json.loads((d / "generation_config.json").read_text()) == {"eos_token_id": 42}

    def test_reads_ids_from_a_nested_text_config(self, repo):
        """A vision-language config keeps them under text_config."""
        d = self._model(repo, "vlm", {"text_config": _UITARS_IDS})
        assert start._ensure_generation_config("vlm")
        written = json.loads((d / "generation_config.json").read_text())
        assert written["eos_token_id"] == 151645

    def test_eos_may_be_a_list(self, repo):
        d = self._model(repo, "vlm", {"eos_token_id": [151645, 151643]})
        assert start._ensure_generation_config("vlm")
        written = json.loads((d / "generation_config.json").read_text())
        assert written["eos_token_id"] == [151645, 151643]

    def test_gives_up_when_there_is_no_config_to_read(self, repo):
        self._model(repo, "vlm", None)
        assert not start._ensure_generation_config("vlm")

    def test_gives_up_when_the_config_has_no_eos(self, repo):
        self._model(repo, "vlm", {"vocab_size": 152064})
        assert not start._ensure_generation_config("vlm")

    def test_missing_folder_is_not_a_crash(self, repo):
        assert not start._ensure_generation_config("never-exported")

    def test_unreadable_config_is_not_a_crash(self, repo):
        d = self._model(repo, "vlm", None)
        (d / "config.json").write_text("{ this is not json")
        assert not start._ensure_generation_config("vlm")


# ── Compiled-model cache ──────────────────────────────────────────────────────
# Without CACHE_DIR the GPU plugin recompiles both models from IR on every start,
# which is most of the wait before the app is usable.

class TestCacheDir:

    def _patch(self, plugin_config: str) -> str:
        import re
        return re.sub(r"plugin_config:\s*'([^']*)'",
                      start._with_cache_dir,
                      f"plugin_config: '{plugin_config}'")

    def test_cache_dir_is_added_to_an_empty_config(self, repo):
        out = self._patch("{}")
        assert '"CACHE_DIR"' in out
        assert start._ovms_cache_dir() in out

    def test_existing_settings_survive(self, repo):
        """plugin_config already carries device and decoding settings."""
        out = self._patch('{"target_device": "GPU", "prompt_lookup": true}')
        assert '"target_device": "GPU"' in out
        assert '"prompt_lookup": true' in out
        assert '"CACHE_DIR"' in out

    def test_rewriting_twice_changes_nothing(self, repo):
        """_sync_servable runs on every start; it must not report a fake change."""
        once = self._patch("{}")
        import re
        twice = re.sub(r"plugin_config:\s*'([^']*)'", start._with_cache_dir, once)
        assert once == twice

    def test_unparseable_config_is_left_alone(self, repo):
        out = self._patch("{not json")
        assert out == "plugin_config: '{not json'"

    def test_path_has_no_backslashes(self, repo):
        r"""A Windows '\' would not survive protobuf unescaping into valid JSON."""
        assert "\\" not in start._ovms_cache_dir()

    def test_the_written_config_is_valid_json(self, repo):
        """The whole point: OVMS has to be able to parse it back."""
        import re
        out = self._patch('{"target_device": "GPU"}')
        payload = re.search(r"plugin_config: '([^']*)'", out).group(1)
        assert json.loads(payload)["CACHE_DIR"] == start._ovms_cache_dir()
