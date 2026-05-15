"""Smoke tests for the ``morseformer`` console CLI.

Avoid loading models or hitting the Hub — these tests only exercise
argument parsing, the preset table, and the local-resolution branch of
the model registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from morseformer import __version__
from morseformer.cli import build_parser, main
from morseformer.cli.presets import DEFAULT_PRESET, PRESETS, get_preset
from morseformer.cli.registry import (
    RECOMMENDED_ACOUSTIC,
    REGISTRY,
    known_names,
    resolve_model,
)


def test_version_string_matches_package() -> None:
    import re

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
    assert m is not None, "could not parse version from pyproject.toml"
    assert __version__ == m.group(1), (
        f"morseformer.__version__ = {__version__!r} but pyproject.toml has "
        f"version = {m.group(1)!r}; keep them in sync."
    )


def test_top_level_parser_has_expected_subcommands() -> None:
    parser = build_parser()
    # argparse exposes choices on the subparsers action.
    sub_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    assert set(sub_action.choices) == {"decode", "live", "gui", "models"}


def test_version_flag_prints_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_models_list_default_only_shows_recommended(capsys) -> None:
    rc = main(["models", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rnnt_phase5_5" in out
    assert "lm_phase5_2" in out
    # Demoted at v0.6.2 — kept in registry but no longer recommended.
    assert "rnnt_phase5_8" not in out
    # Legacy 46-vocab models must be hidden by default.
    assert "rnnt_phase3_0" not in out
    assert "lm_phase4_0" not in out


def test_models_list_advanced_shows_legacy(capsys) -> None:
    rc = main(["models", "list", "--advanced"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rnnt_phase3_0" in out
    assert "lm_phase4_0" in out


def test_known_names_recommended_only() -> None:
    rec = known_names(advanced=False)
    assert "rnnt_phase5_5" in rec
    assert "lm_phase5_2" in rec
    assert "rnnt_phase3_0" not in rec
    assert "rnnt_phase5_7" not in rec  # demoted to legacy in v0.6.0
    assert "rnnt_phase5_8" not in rec  # demoted at v0.6.2 (bench LCWO v1)


def test_known_names_advanced_includes_all() -> None:
    assert set(known_names(advanced=True)) == set(REGISTRY)


def test_recommended_acoustic_is_phase5_5() -> None:
    assert RECOMMENDED_ACOUSTIC == "rnnt_phase5_5"


def test_default_preset_is_live() -> None:
    assert DEFAULT_PRESET == "live"


def test_all_four_presets_present() -> None:
    assert set(PRESETS) == {"live", "prose", "contest", "conservative"}


def test_live_preset_has_v0_6_2_defaults() -> None:
    p = get_preset("live")
    assert p.acoustic == "rnnt_phase5_5"
    assert p.confidence_threshold == 0.6
    assert p.digit_threshold == 0.90
    assert p.lm is None


def test_prose_preset_enables_fusion() -> None:
    p = get_preset("prose")
    assert p.lm == "lm_phase5_2"
    assert p.fusion_weight == 0.7


def test_contest_preset_loosens_thresholds() -> None:
    p = get_preset("contest")
    assert p.confidence_threshold < get_preset("live").confidence_threshold
    assert p.digit_threshold < get_preset("live").digit_threshold


def test_conservative_preset_tightens_thresholds() -> None:
    p = get_preset("conservative")
    assert p.confidence_threshold > get_preset("live").confidence_threshold
    assert p.digit_threshold > get_preset("live").digit_threshold


def test_get_preset_unknown_raises() -> None:
    with pytest.raises(SystemExit):
        get_preset("does-not-exist")


def test_resolve_model_finds_release_dir(tmp_path: Path) -> None:
    """When release/<file>.pt exists, resolve_model returns it."""
    release = tmp_path / "release"
    release.mkdir()
    fake = release / "rnnt_phase5_8.pt"
    fake.write_bytes(b"")
    found = resolve_model("rnnt_phase5_8", repo_root=tmp_path)
    assert found == fake


def test_resolve_model_falls_back_to_checkpoints(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints" / "phase5_8"
    ckpt.mkdir(parents=True)
    fake = ckpt / "last.pt"
    fake.write_bytes(b"")
    found = resolve_model("rnnt_phase5_8", repo_root=tmp_path)
    assert found == fake


def test_resolve_model_unknown_name_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        resolve_model("rnnt_phase999", repo_root=tmp_path)


def test_decode_subcommand_help_does_not_crash(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["decode", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "preset" in out


def test_live_subcommand_help_does_not_crash(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["live", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "preset" in out
