"""The tracked files under configs/ agree with each other and with the config generator."""

import generate_forecast_configs
import pandas as pd


def test_pretraining_corpus_is_water_disjoint_from_the_evaluation_waters(repo_root) -> None:
    """The 441 corpus waters are candidates from the selection, never an evaluation water."""
    configs = repo_root / "configs"
    corpus = (configs / "pretraining_waters.txt").read_text().split()
    candidates = set(pd.read_csv(configs / "pretraining_corpus.csv").water_id)
    evaluation_ids = set(pd.read_csv(configs / "target_waters.csv").hydrolakes_id.dropna())
    assert len(corpus) == len(set(corpus)) == 441
    assert set(corpus) <= candidates
    assert not set(corpus) & evaluation_ids


def test_paper_configs_are_what_the_generator_writes(tmp_path, repo_root) -> None:
    """configs/paper/ holds exactly the generator's configs of the main comparison."""
    generate_forecast_configs.generate(tmp_path / "generated", tmp_path / "paper")
    tracked = repo_root / "configs" / "paper"
    names = sorted(path.stem for path in tracked.glob("*.yaml"))
    assert names == sorted(generate_forecast_configs.PAPER_CONFIGS)
    for name in names:
        written = (tmp_path / "paper" / f"{name}.yaml").read_text()
        assert written == (tracked / f"{name}.yaml").read_text(), name
        assert written == (tmp_path / "generated" / f"{name}.yaml").read_text(), name
