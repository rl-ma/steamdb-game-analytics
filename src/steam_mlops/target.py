"""The target: is a game "lasting"?

A game lasts if it has more current players than expected for a game of its size (total
reviews) and release year, in the top 25% of its year. The expected-players regression and the
cutoffs are fitted on the training set only and then applied to the test set.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from steam_mlops import config

YEARS = list(range(config.WINDOW_START, config.WINDOW_END + 1))


def _size_and_year(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"log_reviews": np.log10(df["total_reviews"])}, index=df.index)
    for year in YEARS[1:]:
        out[f"year_{year}"] = (df["release_year"] == year).astype(float)
    return out


def _log_players(df: pd.DataFrame) -> pd.Series:
    return np.log10(df["peak_ccu"] + 1)


@dataclass
class LastingLabeler:
    """Fit on training games, then label any games with .label()."""

    quantile: float = config.LASTING_QUANTILE

    def fit(self, train: pd.DataFrame) -> "LastingLabeler":
        self.regression_ = LinearRegression().fit(_size_and_year(train), _log_players(train))
        residual = self.residual(train)
        self.cutoffs_ = residual.groupby(train["release_year"]).quantile(self.quantile)
        return self

    def residual(self, df: pd.DataFrame) -> pd.Series:
        """log10(players + 1) minus what the regression expects for that size and year."""
        return _log_players(df) - self.regression_.predict(_size_and_year(df))

    def label(self, df: pd.DataFrame) -> np.ndarray:
        return (self.residual(df) > df["release_year"].map(self.cutoffs_)).astype(int).to_numpy()

    @property
    def slope(self) -> float:
        """How current players scale with reviews (1.23 means 10x reviews -> ~17x players)."""
        return float(self.regression_.coef_[0])
