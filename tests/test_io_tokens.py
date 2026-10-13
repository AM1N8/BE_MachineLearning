"""Parser regressions, including the complete available event vocabulary."""

import csv
import gzip
import unicodedata
from pathlib import Path

import pytest

from copilote.io import file_hash, read_sessions, write_json
from copilote.tokens import filled_context, parse_token


@pytest.mark.parametrize("wrapped", [False, True])
def test_ragged_quotes(tmp_path, wrapped):
    text = "abc,Firefox,t5,Saisie dans un champ(MAINT)<DEF_1>$VT$1"
    path = tmp_path / "train.csv"
    path.write_text(('"' + text + '"' if wrapped else text) + "\nxyz,Opera,t10\n")
    rows = read_sessions(path, labelled=True)
    assert rows[0].label == "abc"
    assert len(rows[0].tokens) == 2
    assert rows[1].row_id == 1
    assert len(file_hash(path)) == 64


def test_csv_quoted_comma_gzip(tmp_path):
    path = tmp_path / "compressed"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["Firefox", "Action, avec virgule", "t5"])
    row = read_sessions(path, labelled=False)[0]
    assert row.tokens[0] == "Action, avec virgule"
    assert row.label is None


@pytest.mark.parametrize("text", ["", "\n", "abc\n", "abc,,t5\n", "abc,Firefox,,t5\n"])
def test_invalid_rows(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_sessions(path, labelled=True)


def test_grammar():
    raw = "Saisie dans un champ(A)(B)<DEF_1>$VT$1"
    token = parse_token(raw)
    assert token.action == "Saisie dans un champ"
    assert token.screens == ("A", "B")
    assert token.configs == ("DEF_1",)
    assert token.suffix_one
    assert token.reconstruct() == raw
    nested = "Tomcat(fr.Controller.method (Controller.java:514))"
    assert parse_token(nested).screens == ("fr.Controller.method (Controller.java:514)",)
    assert parse_token(nested).reconstruct() == nested
    assert parse_token("t11790").seconds == 11790
    assert parse_token("Action 12").action == "Action 12"
    assert parse_token("Sélection d’un onglet") != parse_token("Sélection d'un onglet")
    assert parse_token("E\u0301cran").reconstruct() == "Écran"
    for invalid in ["", "Action(A", "(A)"]:
        with pytest.raises(ValueError):
            parse_token(invalid)


def test_context_is_session_local():
    rows = filled_context(["Action(A)<C>$X$1", "t5", "Saisie1", "Fermeture de session", "Action"])
    assert rows[1][1:] == (("A",), ("C",), ("X",))
    assert rows[-1][1:] == ((), (), ())
    assert filled_context(["Action"])[0][1:] == ((), (), ())
    assert filled_context(["Action(A)", "Démarrage serveur", "Action"])[-1][1] == ()


def test_json_rejects_nan(tmp_path):
    write_json(tmp_path / "ok.json", {"value": 3})
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_all_available_tokens():
    root = Path(__file__).resolve().parents[1] / "data"
    root = root / "raw" if (root / "raw/train.csv").exists() else root
    if not (root / "train.csv").exists():
        pytest.skip("Private data not installed")
    train = read_sessions(root / "train.csv", labelled=True)
    test = read_sessions(root / "test.csv", labelled=False)
    assert (len(train), len(test), len({row.label for row in train})) == (3279, 324, 247)
    for raw in {token for row in train + test for token in row.tokens}:
        assert parse_token(raw).reconstruct() == unicodedata.normalize("NFC", raw)
