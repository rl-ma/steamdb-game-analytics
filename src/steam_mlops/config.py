"""Project-wide settings. Everything that might change lives here, not inside the code."""

import os
from pathlib import Path

# ---------------------------------------------------------------- data
# Raw files are downloaded here once (they are ~600 MB, so never commit them).
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"

# Pinned dataset version (Hugging Face commit of 2 October 2026), so every run uses the same data.
DATASET_REVISION = "ba4e26785af33bee500e96597068c6e77f4edea9"
_BASE_URL = (
    f"https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/{DATASET_REVISION}/"
)
RAW_FILES = {
    "steam_games.parquet": _BASE_URL
    + "data/train-00000-of-00001.parquet",  # main data (41 columns)
    "steam_games.csv": _BASE_URL + "games.csv",  # adds Discount and Tags
}

# ---------------------------------------------------------------- problem definition
WINDOW_START, WINDOW_END = 2019, 2023  # release years we study
HIT_THRESHOLD = 500  # total reviews needed to count as a "hit"
LASTING_QUANTILE = 0.75  # "lasting" = top 25% above expected players, within each release year

# ---------------------------------------------------------------- evaluation
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------- MLflow
# Locally this is a SQLite file; in docker compose set MLFLOW_TRACKING_URI=http://mlflow:5000.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "steam-staying-power")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "steam-staying-power")
CHAMPION_ALIAS = "champion"  # the API loads models:/steam-staying-power@champion

# ---------------------------------------------------------------- domain lists
GAME_GENRES = {
    "Action",
    "Adventure",
    "Casual",
    "Indie",
    "RPG",
    "Simulation",
    "Strategy",
    "Sports",
    "Racing",
    "Massively Multiplayer",
    "Free To Play",
    "Early Access",
    "Violent",
    "Gore",
}
IDLE_TAGS = {"Idler", "Clicker", "Incremental"}

# Category values added to store pages after launch (leakage). Valve introduced the
# accessibility labels in 2025, after every game in our window had launched.
LEAKY_CATEGORIES = {
    "Steam Trading Cards",
    "SteamVR Collectibles",
    "Steam Timeline",
    "Adjustable Text Size",
    "Subtitle Options",
    "Color Alternatives",
    "Camera Comfort",
    "Custom Volume Controls",
    "Stereo Sound",
    "Surround Sound",
    "Narrated Game Menus",
    "Chat Speech-to-text",
    "Chat Text-to-speech",
    "Playable without Timed Input",
    "Keyboard Only Option",
    "Mouse Only Option",
    "Touch Only Option",
    "Adjustable Difficulty",
    "Save Anytime",
    "HDR available",
}

SOCIAL_CATEGORIES = {
    "Multi-player",
    "Online Co-op",
    "Co-op",
    "Online PvP",
    "PvP",
    "Cross-Platform Multiplayer",
    "Shared/Split Screen",
    "Shared/Split Screen Co-op",
    "Shared/Split Screen PvP",
    "LAN Co-op",
    "LAN PvP",
    "MMO",
    "Remote Play Together",
    "Steam Workshop",
    "Includes level editor",
}
