import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────── UTILITIES ──────────────── #

def _to_datetime(df, col):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df.dropna(subset=[col])


def _filter_date_range(df, date_col, start_date=None, end_date=None):
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    return df


def resample_monthly(df, date_col, agg_dict):
    return df.set_index(date_col).resample("ME").agg(agg_dict).reset_index()


def resample_weekly_avg(df, date_col, value_col, smooth_window=1):
    weekly = df.set_index(date_col).resample("W").mean(numeric_only=True)
    if smooth_window > 1:
        weekly = weekly.rolling(window=smooth_window, center=True, min_periods=1).mean()
    return weekly.reset_index()


# ──────────────── PRICE FUNCTIONS ──────────────── #

def prepare_price_monthly(df, date_col="DateTime", price_col="Final price", low_col="Historical Low",
                          agg="min", start_date=None, end_date=None):
    df = df.copy()
    df = _to_datetime(df, date_col)
    df = df.dropna(subset=[price_col, low_col])
    df = _filter_date_range(df, date_col, start_date, end_date)
    return resample_monthly(df, date_col, {price_col: agg, low_col: agg})


def plot_monthly_prices(df, date_col="DateTime", price_col="Final price", low_col="Historical Low",
                        years=None, rolling=0):
    df = df.copy()
    if years:
        years = [years] if isinstance(years, int) else years
        df = df[df[date_col].dt.year.isin(years)]

    if rolling > 1:
        df[price_col] = df[price_col].rolling(rolling, center=True).mean()
        df[low_col] = df[low_col].rolling(rolling, center=True).mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=date_col, y=price_col, label="Final Price")
    sns.lineplot(data=df, x=date_col, y=low_col, label="Historical Low")
    plt.title(f"Monthly Prices{f' – Years: {years}' if years else ''}{f' (Smoothed {rolling}-mo)' if rolling > 1 else ''}")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ──────────────── PLAYER FUNCTIONS ──────────────── #

def prepare_players_monthly(df, date_col="DateTime", players_col="Players",
                            agg="mean", start_date=None, end_date=None):
    df = df.copy()
    df = _to_datetime(df, date_col)
    df = df.dropna(subset=[players_col])
    df = _filter_date_range(df, date_col, start_date, end_date)
    return resample_monthly(df, date_col, {players_col: agg})


def plot_daily_players(df, date_col="DateTime", players_col="Players",
                       start_date=None, end_date=None, rolling=7):
    df = df.copy()
    df = _to_datetime(df, date_col)
    df = df.dropna(subset=[players_col])
    df = _filter_date_range(df, date_col, start_date, end_date)

    df["Smoothed"] = df[players_col].rolling(rolling, center=True).mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=date_col, y=players_col, label="Daily", alpha=0.3)
    sns.lineplot(data=df, x=date_col, y="Smoothed", label=f"{rolling}-Day Avg", color='red')
    plt.title("Daily Players with Rolling Average")
    plt.xlabel("Date")
    plt.ylabel("Players")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_weekly_players(df, date_col="DateTime", players_col="Players",
                        start_date=None, end_date=None):
    df = df.copy()
    df = _to_datetime(df, date_col)
    df = df.dropna(subset=[players_col])
    df = _filter_date_range(df, date_col, start_date, end_date)
    weekly = resample_weekly_avg(df, date_col, players_col)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=weekly, x=date_col, y=players_col, label="Weekly Avg", color='green')
    plt.title("Weekly Average Player Counts")
    plt.xlabel("Date")
    plt.ylabel("Players")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ──────────────── COMBO PLOTS ──────────────── #

def plot_prices_and_players_side_by_side(raw_price_df, raw_players_df,
                                         years=range(2016, 2026),
                                         price_date="DateTime", price_col="Final price",
                                         low_col="Historical Low", players_date="DateTime",
                                         players_col="Players", price_agg="min"):

    n = len(years)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))

    if n == 1:
        axes = axes.reshape(1, 2)

    raw_price_df = _to_datetime(raw_price_df.copy(), price_date)
    raw_players_df = _to_datetime(raw_players_df.copy(), players_date)

    for i, year in enumerate(years):
        prices = prepare_price_monthly(
            raw_price_df, price_date, price_col, low_col,
            agg=price_agg, start_date=f"{year}-01-01", end_date=f"{year}-12-31"
        )

        players = raw_players_df[
            (raw_players_df[players_date].dt.year == year) & raw_players_df[players_col].notna()
        ].copy()

        weekly = resample_weekly_avg(players, players_date, players_col)

        sns.lineplot(data=prices, x=price_date, y=price_col, label="Final", ax=axes[i, 0])
        sns.lineplot(data=prices, x=price_date, y=low_col, label="Low", ax=axes[i, 0])
        axes[i, 0].set_title(f"{year} Prices")
        axes[i, 0].tick_params(axis='x', rotation=45)

        sns.lineplot(data=weekly, x=players_date, y=players_col, color="green", ax=axes[i, 1])
        axes[i, 1].set_title(f"{year} Weekly Avg Players")
        axes[i, 1].tick_params(axis='x', rotation=45)

    for ax in axes[-1, :]:
        ax.set_xlabel("Date")
    plt.tight_layout()
    plt.show()


def plot_combined_price_players(raw_price_df, raw_players_df, year,
                                price_date="DateTime", price_col="Final price", low_col="Historical Low",
                                players_date="DateTime", players_col="Players",
                                price_agg="min", smooth_players=True, smooth_window=1):

    prices = prepare_price_monthly(
        raw_price_df, price_date, price_col, low_col, agg=price_agg,
        start_date=f"{year}-01-01", end_date=f"{year}-12-31"
    )

    players = _to_datetime(raw_players_df.copy(), players_date)
    players = players[
        (players[players_date].dt.year == year) & players[players_col].notna()
    ].copy()

    weekly = resample_weekly_avg(players, players_date, players_col, smooth_window if smooth_players else 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.lineplot(data=prices, x=price_date, y=price_col, ax=ax1, color="#00FFFF", label="Final Price", linewidth=2)
    sns.lineplot(data=prices, x=price_date, y=low_col, ax=ax1, color="#9B59B6", label="Historical Low", linestyle="--", linewidth=2)
    ax1.set_ylabel("Price ($)", color="#00FFFF")
    ax1.tick_params(axis="y", labelcolor="#00FFFF")
    ax1.set_xlabel("Date")
    ax1.set_title(f"{year} Prices and Player Counts")

    ax2 = ax1.twinx()
    sns.lineplot(data=weekly, x=players_date, y=players_col, ax=ax2, color="#2ECC71", label="Players", linewidth=2)
    ax2.set_ylabel("Players", color="#2ECC71")
    ax2.tick_params(axis="y", labelcolor="#2ECC71")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
