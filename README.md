# Steam Game Analytics – Price and Player Behavior

This project analyzes historical **price and player count data** for six popular games on Steam, using publicly available records from [SteamDB.info](https://steamdb.info). It is part of a broader effort to explore how pricing strategies influence user engagement over time in live-service and multiplayer games.

## Dataset

The dataset includes:
- **price history** (final price and historical low)
- **Daily concurrent player counts**

The data spans multiple years for six different titles, all of which have had active player communities and frequent price changes on Steam. Games were selected based on their relevance, longevity, and variety in sales strategies.

## Objective

The core goal of this analysis is to identify patterns and relationships between **price movements** (e.g., discounts, price hikes) and **player behavior** (e.g., activity spikes, retention trends). This can provide valuable insight into player sensitivity to pricing, the effectiveness of discounting strategies, and the dynamics of long-term engagement.

## Case Study: Rainbow Six Siege

While data for six games is included in this repository, the notebook focuses on **Rainbow Six Siege** as a detailed case study. The game was chosen due to the pronounced price fluctuations observed during 2023, which allowed for a clear and meaningful analysis of how pricing shifts correlate with player response. The methods and visualizations developed here can be applied to the remaining titles with minimal adjustment.

## Project Structure

This repository is organized to clearly separate reusable code, analysis notebooks, and raw data for ease of exploration and extension.

### Code

- **`utils.py`**  
  Contains general-purpose data processing functions such as data cleaning, and aggregation utilities. Designed for reuse across different games and analyses.

- **`plots.py`**  
  Includes plotting functions to create standardized visualizations of price and player metrics. Handles styling, dual-axis charts, smoothing, and comparative views.

- **`visualization.ipynb`**  
  A Jupyter Notebook demonstrating an end-to-end exploratory analysis focused on *Rainbow Six Siege*. It showcases data loading and visualization using the utilities above, providing a comprehensive case study.

### Data

- **`/data/`**  
  The root data folder contains subdirectories named after each game in the dataset.

  Each game folder includes two CSV files:
  - `prices.csv`: Historical price data including final price and historical lows.
  - `player_counts.csv`: Daily concurrent player count data.

This structure allows straightforward access to each game’s data, facilitating comparative analysis or applying the pipeline to additional titles.

## Skills Demonstrated

- Time-series aggregation and resampling (daily, weekly, monthly)
- Exploratory data visualization using Seaborn and Matplotlib
- Comparative analysis and dual-axis plotting
- Clean, modular code for reuse across datasets
- Narrative explanation of real-world user behavior using data

## Next Steps

This framework can be extended to:
- Compare sales events across games (e.g., Steam Summer Sale impact)
- Add update/release annotations to improve behavioral context
- Perform basic forecasting or clustering of engagement trends

---

This project demonstrates practical data visualization and analysis in a game analytics context, combining technical clarity with storytelling to support decision-making and insight generation.
