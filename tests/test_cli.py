from __future__ import annotations

from spectrumov import cli


def _encoder_action_choices(parser) -> list[str]:
    action = next(a for a in parser._actions if a.dest == "encoder")
    return list(action.choices)


def test_build_parser_includes_pyav_when_available(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_pyav_available", lambda: True)
    parser = cli.build_parser()
    assert _encoder_action_choices(parser) == ["ffmpeg", "pyav", "opencv"]


def test_build_parser_excludes_pyav_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_pyav_available", lambda: False)
    parser = cli.build_parser()
    assert _encoder_action_choices(parser) == ["ffmpeg", "opencv"]

