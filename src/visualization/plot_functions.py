import os
import sys

import ipywidgets as widgets
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import interact
from matplotlib.widgets import Button
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.constants import POPULATION_DATA, NUMBER_OF_DAYS_OF_WEEK, NUMBER_OF_DAYS_IN_MONTH, NUMBER_OF_DAYS_OF_MONTH


def shop_sales_dynamics_plot(train, shops):
    """
    Interactive plot of average daily sales for a selected shop.
    """

    shop_names = shops["shop_name"].unique()

    data_for_shop_plots = pd.merge(
        train[
            [
                "item_cnt_day",
                "date_block_num",
                "month_day",
                "week_day",
                "shop_id",
            ]
        ],
        shops[["shop_id", "shop_name"]],
        on="shop_id",
        how="left",
    )

    def plot_single_shop(shop, period):
        plt.close("all")
        plt.figure(figsize=(15, 5))

        shop_data = data_for_shop_plots[
            data_for_shop_plots["shop_name"] == shop
        ]

        if period == "By months (date_block_num)":
            group_col = "date_block_num"
            x_label = "Month (date_block_num)"
            title = f"Daily Average Sales Dynamics by Month for Shop: {shop}"
            mapping_dict = NUMBER_OF_DAYS_IN_MONTH
            is_string_key = True

        elif period == "By days of week":
            group_col = "week_day"
            x_label = "Day of the week"
            title = f"Daily Average Sales Dynamics by Day of Week for Shop: {shop}"
            mapping_dict = NUMBER_OF_DAYS_OF_WEEK
            is_string_key = False

        elif period == "By days of month":
            group_col = "month_day"
            x_label = "Day of the month"
            title = f"Daily Average Sales Dynamics by Day of Month for Shop: {shop}"
            mapping_dict = NUMBER_OF_DAYS_OF_MONTH
            is_string_key = False

        agg_data = (
            shop_data.groupby(group_col)["item_cnt_day"]
            .sum()
            .reset_index(name="total_sales")
        )

        if is_string_key:
            days_count = agg_data[group_col].astype(str).map(mapping_dict)
        else:
            days_count = agg_data[group_col].map(mapping_dict)

        agg_data["normalized_sales"] = (
            agg_data["total_sales"] / days_count
        )

        sns.lineplot(
            data=agg_data,
            x=group_col,
            y="normalized_sales",
            marker="o",
            color="g",
            label="Average daily sales",
        )

        if period == "By months (date_block_num)":
            plt.xlim(
                data_for_shop_plots["date_block_num"].min(),
                data_for_shop_plots["date_block_num"].max(),
            )

        elif period == "By days of week":
            if agg_data[group_col].min() == 0:
                plt.xticks(
                    range(7),
                    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                )
                plt.xlim(0, 6)
            else:
                plt.xticks(
                    range(1, 8),
                    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                )
                plt.xlim(1, 7)

        elif period == "By days of month":
            plt.xticks(range(1, 32))
            plt.xlim(1, 31)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Average items sold per day")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    interact(
        plot_single_shop,
        shop=shop_names,
        period=widgets.Dropdown(
            options=[
                "By months (date_block_num)",
                "By days of week",
                "By days of month",
            ],
            value="By months (date_block_num)",
            description="Timeframe:",
        ),
    )





def plot_price_dynamics_in_the_shop_for_item(df, shop_id, item_id):

    result = df[
        (df["shop_id"] == shop_id)
        & (df["item_id"] == item_id)
    ].sort_values("date")

    plt.figure(figsize=(12, 6))

    plt.plot(
        result["date"],
        result["item_price"],
        marker="o",
        linestyle="-",
        color="b",
        markersize=4,
    )

    plt.title(
        f"Price dynamics (Shop {shop_id}, Item {item_id})",
        fontsize=14,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.7)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()









def plot_average_price_dynamics_in_the_shop_for_category(
    df,
    shop_id,
    categories,
    start_date,
    end_date,
    highlight_date=None,
):
    """
    Plot average daily price dynamics for selected categories in a shop.

    Parameters
    ----------
    df : pd.DataFrame
    shop_id : int
    categories : list[str]
        List of main categories.
    start_date : str or datetime
    end_date : str or datetime
    highlight_date : str or datetime, optional
        Date to highlight on the plot.
    """

    mask = (
        (df["shop_id"] == shop_id)
        & (df["main_category"].isin(categories))
        & (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    )

    result = (
        df.loc[mask]
        .groupby("date")["item_price"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        result["date"],
        result["item_price"],
        marker="o",
        linestyle="-",
        color="b",
        label="Average price",
    )

    if highlight_date is not None:
        highlight_date = pd.to_datetime(highlight_date)

        price = result.loc[
            result["date"] == highlight_date,
            "item_price",
        ]

        if not price.empty:
            plt.scatter(
                highlight_date,
                price.iloc[0],
                color="red",
                s=100,
                zorder=5,
                label=highlight_date.strftime("%Y-%m-%d"),
            )

            plt.annotate(
                f"Average price: {price.iloc[0]:.2f}",
                xy=(highlight_date, price.iloc[0]),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="red"),
            )

    plt.title("Average price dynamics in the shop per day")
    plt.xlabel("Date")
    plt.ylabel("Average price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()





def plot_price_vs_sales(df):
    """
    Scatter plot for outlier detection: item price vs sales volume.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'item_cnt_day' and 'item_price' columns.
    """

    plt.figure(figsize=(10, 4))

    sns.scatterplot(
        data=df,
        x="item_cnt_day",
        y="item_price",
    )

    plt.title("Outlier Detection: Price vs. Sales Volume")
    plt.xlabel("Sales volume")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()




def plot_ems_price_dynamics(df):
    """
    Plot EMS delivery price dynamics over time.
    """

    ems_data = df[df["item_name"] == "Доставка (EMS)"].copy()

    ems_data["date"] = pd.to_datetime(ems_data["date"])

    daily_sales = (
        ems_data.groupby("date")["item_price"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        daily_sales.index,
        daily_sales.values,
        marker="o",
        linestyle="-",
        color="b",
    )

    plt.title("Price dynamics: Доставка (EMS)")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_item_sales_dynamics(df, item_name):
    """
    Plot daily sales dynamics for a selected item.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'item_name', 'date' and 'item_cnt_day'.
    item_name : str
        Name of the item.
    """

    item_data = df[df["item_name"] == item_name].copy()

    item_data["date"] = pd.to_datetime(item_data["date"])

    daily_sales = (
        item_data.groupby("date")["item_cnt_day"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        daily_sales.index,
        daily_sales.values,
        marker="o",
        linestyle="-",
        color="b",
    )

    plt.title(f"Sales dynamics: {item_name}")
    plt.xlabel("Date")
    plt.ylabel("Items sold")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_total_sales(df):
    """
    Interactive plot of average daily sales.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing:
        item_cnt_day, date_block_num, week_day, month_day.
    """

    def _plot(period):
        plt.close("all")
        plt.figure(figsize=(15, 5))

        if period == "By months (date_block_num)":
            group_col = "date_block_num"
            x_label = "Month (date_block_num)"
            title = "Average Daily Sales by Month"
            mapping_dict = NUMBER_OF_DAYS_IN_MONTH
            is_string_key = True

        elif period == "By days of week":
            group_col = "week_day"
            x_label = "Day of the week"
            title = "Average Daily Sales by Day of Week"
            mapping_dict = NUMBER_OF_DAYS_OF_WEEK
            is_string_key = False

        elif period == "By days of month":
            group_col = "month_day"
            x_label = "Day of the month"
            title = "Average Daily Sales by Day of Month"
            mapping_dict = NUMBER_OF_DAYS_OF_MONTH
            is_string_key = False

        else:
            raise ValueError(f"Unknown period: {period}")

        agg_data = (
            df.groupby(group_col)["item_cnt_day"]
            .sum()
            .reset_index(name="total_sales")
        )

        if is_string_key:
            days_count = agg_data[group_col].astype(str).map(mapping_dict)
        else:
            days_count = agg_data[group_col].map(mapping_dict)

        agg_data["normalized_sales"] = (
            agg_data["total_sales"] / days_count
        )

        sns.lineplot(
            data=agg_data,
            x=group_col,
            y="normalized_sales",
            marker="o",
            color="b",
            label="Average daily sales",
        )

        if period == "By months (date_block_num)":
            plt.xlim(
                df["date_block_num"].min(),
                df["date_block_num"].max(),
            )

        elif period == "By days of week":
            if agg_data[group_col].min() == 0:
                plt.xticks(
                    range(7),
                    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                )
                plt.xlim(0, 6)
            else:
                plt.xticks(
                    range(1, 8),
                    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                )
                plt.xlim(1, 7)

        elif period == "By days of month":
            plt.xticks(range(1, 32))
            plt.xlim(1, 31)

        plt.title(title, fontsize=14)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("Average items sold per day", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    interact(
        _plot,
        period=widgets.Dropdown(
            options=[
                "By months (date_block_num)",
                "By days of week",
                "By days of month",
            ],
            value="By months (date_block_num)",
            description="Timeframe:",
        ),
    )


def plot_autocorrelation(df):
    """
    Interactive autocorrelation plot for total sales.
    """

    def _plot(period):
        plt.close("all")
        fig, ax = plt.subplots(figsize=(16, 6))

        if period == "Monthly (33 lags)":
            sales = (
                df.groupby(df["date"].dt.to_period("M"))["item_cnt_day"]
                .sum()
            )
            lags = 33
            title = "Autocorrelation of Total Monthly Sales"
            xlabel = "Lag (month)"

        elif period == "Weekly (147 lags)":
            sales = (
                df.groupby(df["date"].dt.to_period("W"))["item_cnt_day"]
                .sum()
            )
            lags = 147
            title = "Autocorrelation of Total Weekly Sales"
            xlabel = "Lag (week)"

        elif period == "Weekly (20 lags)":
            sales = (
                df.groupby(df["date"].dt.to_period("W"))["item_cnt_day"]
                .sum()
            )
            lags = 20
            title = "Autocorrelation of Total Weekly Sales"
            xlabel = "Lag (week)"

        else:
            raise ValueError(f"Unknown period: {period}")

        plot_acf(sales, lags=lags, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Correlation coefficient")
        ax.grid(True)

        plt.show()

    interact(
        _plot,
        period=widgets.Dropdown(
            options=[
                "Monthly (33 lags)",
                "Weekly (147 lags)",
                "Weekly (20 lags)",
            ],
            value="Monthly (33 lags)",
            description="Period:",
        ),
    )




def plot_stl_decomposition(df, period=365):
    """
    Plot STL decomposition of daily sales.
    """

    daily_sales = (
        df.groupby("date")["item_cnt_day"]
        .sum()
    )

    daily_sales.index = pd.to_datetime(daily_sales.index)
    daily_sales = daily_sales.asfreq("D").fillna(0)

    stl = STL(daily_sales, period=period, robust=True)
    result = stl.fit()

    fig = result.plot()
    fig.set_size_inches(12, 10)

    plt.tight_layout()
    plt.show()




def plot_sales_by_shop(df):
    """
    Plot total sales by shop.
    """

    shop_sales = (
        df.groupby("shop_name")["item_cnt_day"]
        .sum()
        .sort_values()
    )

    plt.figure(figsize=(10, 14))

    shop_sales.plot(
        kind="barh",
        color="skyblue",
        edgecolor="gray",
    )

    plt.title("Items sold per shop")
    plt.xlabel("Items sold")
    plt.ylabel("Shop")

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()




def plot_sales_by_city(df):
    """
    Plot total sales by city.
    """

    city_sales = (
        df.groupby("city")["item_cnt_day"]
        .sum()
        .sort_values()
    )

    plt.figure(figsize=(10, 14))

    city_sales.plot(
        kind="barh",
        color="skyblue",
        edgecolor="gray",
    )

    plt.title("Sales per city")
    plt.xlabel("Number of items sold")
    plt.ylabel("City")

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()







def plot_sales_per_capita(df):
    city_sales = (
        df[
            ~df["city"].isin(["Интернет", "Выездная торговля"])
        ]
        .groupby("city")["item_cnt_day"]
        .sum()
    )

    df_pop = pd.DataFrame.from_dict(
        POPULATION_DATA,
        orient="index",
        columns=["population"],
    )

    df_metrics = (
        pd.concat([city_sales, df_pop], axis=1)
        .dropna()
    )

    df_metrics["sales_per_capita"] = (
        df_metrics["item_cnt_day"] / df_metrics["population"]
    )

    df_metrics.sort_values(
        "sales_per_capita",
        ascending=False,
        inplace=True,
    )

    plt.figure(figsize=(15, 6))

    df_metrics["sales_per_capita"].plot(
        kind="bar",
        color="coral",
        edgecolor="black",
    )

    plt.title("Number of items sold per person")
    plt.xlabel("City")
    plt.ylabel("Sales per person")

    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()



def plot_top_categories(df, top_n=30):
    """
    Plot top categories by total sales.
    """

    top_categories = (
        df.groupby("main_category")["item_cnt_day"]
        .sum()
        .sort_values()
        .tail(top_n)
    )

    plt.figure(figsize=(10, 12))

    top_categories.plot(
        kind="barh",
        color="lightgreen",
        edgecolor="gray",
    )

    plt.title(f"Top-{top_n} categories based on total sales")
    plt.xlabel("Number of items sold")
    plt.ylabel("Category")

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()




def plot_category_sales(df):
    """
    Interactive plot of average daily sales for a selected category.
    """

    category_names = df["item_category_name"].unique()

    def _plot(category, period):
        plt.close("all")
        plt.figure(figsize=(15, 5))

        category_data = df[df["item_category_name"] == category]

        if period == "By months (date_block_num)":
            group_col = "date_block_num"
            x_label = "Month (date_block_num)"
            title = f"Daily Average Sales Dynamics by Month: {category}"
            mapping_dict = NUMBER_OF_DAYS_IN_MONTH
            is_string_key = True

        elif period == "By days of week":
            group_col = "week_day"
            x_label = "Day of the week"
            title = f"Daily Average Sales Dynamics by Day of Week: {category}"
            mapping_dict = NUMBER_OF_DAYS_OF_WEEK
            is_string_key = False

        elif period == "By days of month":
            group_col = "month_day"
            x_label = "Day of the month"
            title = f"Daily Average Sales Dynamics by Day of Month: {category}"
            mapping_dict = NUMBER_OF_DAYS_OF_MONTH
            is_string_key = False

        else:
            raise ValueError(f"Unknown period: {period}")

        agg_data = (
            category_data.groupby(group_col)["item_cnt_day"]
            .sum()
            .reset_index(name="total_sales")
        )

        if is_string_key:
            days_count = agg_data[group_col].astype(str).map(mapping_dict)
        else:
            days_count = agg_data[group_col].map(mapping_dict)

        agg_data["normalized_sales"] = (
            agg_data["total_sales"] / days_count
        )

        sns.lineplot(
            data=agg_data,
            x=group_col,
            y="normalized_sales",
            marker="o",
            color="b",
            label="Average daily sales",
        )

        if period == "By months (date_block_num)":
            plt.xlim(df["date_block_num"].min(), df["date_block_num"].max())

        elif period == "By days of week":
            if agg_data[group_col].min() == 0:
                plt.xticks(
                    range(7),
                    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                )
                plt.xlim(0, 6)
            else:
                plt.xticks(
                    range(1, 8),
                    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                )
                plt.xlim(1, 7)

        elif period == "By days of month":
            plt.xticks(range(1, 32))
            plt.xlim(1, 31)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Average items sold per day")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    interact(
        _plot,
        category=category_names,
        period=widgets.Dropdown(
            options=[
                "By months (date_block_num)",
                "By days of week",
                "By days of month",
            ],
            value="By months (date_block_num)",
            description="Timeframe:",
        ),
    )




def plot_shop_category_heatmap(df):
    """
    Plot category share in total sales for each shop.
    """

    shop_category_sales = (
        df.groupby(["shop_name", "main_category"])["item_cnt_day"]
        .sum()
        .reset_index()
    )

    shop_total = (
        df.groupby("shop_name")["item_cnt_day"]
        .sum()
        .rename("total_sales")
    )

    shop_category_sales = shop_category_sales.merge(
        shop_total,
        on="shop_name",
    )

    shop_category_sales["percentage"] = (
        shop_category_sales["item_cnt_day"]
        / shop_category_sales["total_sales"]
        * 100
    )

    pivot_df = (
        shop_category_sales
        .pivot(
            index="shop_name",
            columns="main_category",
            values="percentage",
        )
        .fillna(0)
    )

    plt.figure(figsize=(18, 12))

    sns.heatmap(
        pivot_df,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Sales percentage (%)"},
    )

    plt.title("Category share in total sales of each shop")
    plt.xlabel("Main category")
    plt.ylabel("Shop")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()




def plot_cluster_sales_dynamics(df):
    """
    Interactive plot of monthly sales dynamics by category with
    category visibility controls.
    """

    monthly_cat = (
        df.groupby(["date_block_num", "item_category_name"])["item_cnt_day"]
        .sum()
        .unstack(fill_value=0)
    )

    monthly_cat = np.log1p(monthly_cat)

    category_to_cluster = (
        df[["item_category_name", "cluster"]]
        .drop_duplicates()
        .set_index("item_category_name")["cluster"]
        .to_dict()
    )

    clusters_to_categories = {}

    for category, cluster in category_to_cluster.items():
        clusters_to_categories.setdefault(cluster, []).append(category)

    fig, ax = plt.subplots(figsize=(18, 10))

    plt.subplots_adjust(
        left=0.07,
        right=0.35,
        bottom=0.08,
        top=0.98,
    )

    lines = []
    line_by_category = {}

    for category in monthly_cat.columns:
        line, = ax.plot(
            monthly_cat.index,
            monthly_cat[category],
            linewidth=1.5,
            label=category,
        )

        lines.append(line)
        line_by_category[category] = line

    legend = ax.legend(
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        ncol=2,
        fontsize=9,
        handlelength=2.5,
        handletextpad=0.5,
        labelspacing=0.35,
        columnspacing=1.2,
        borderpad=0.4,
        frameon=True,
    )

    legend_mapping = {}

    for legend_line, legend_text, line in zip(
        legend.get_lines(),
        legend.get_texts(),
        lines,
    ):
        legend_line.set_picker(True)
        legend_line.set_pickradius(5)

        legend_text.set_picker(True)

        legend_mapping[legend_line] = line
        legend_mapping[legend_text] = line

    def update_legend():
        for legend_line, legend_text, line in zip(
            legend.get_lines(),
            legend.get_texts(),
            lines,
        ):
            alpha = 1 if line.get_visible() else 0.2
            legend_line.set_alpha(alpha)
            legend_text.set_alpha(alpha)

        fig.canvas.draw_idle()

    def on_pick(event):
        artist = event.artist

        if artist not in legend_mapping:
            return

        line = legend_mapping[artist]
        line.set_visible(not line.get_visible())

        update_legend()

    fig.canvas.mpl_connect("pick_event", on_pick)

    button_ax = plt.axes([0.73, 0.03, 0.23, 0.055])
    toggle_button = Button(button_ax, "Hide / Show all")

    def toggle_all(event):
        visible = not any(line.get_visible() for line in lines)

        for line in lines:
            line.set_visible(visible)

        update_legend()

    toggle_button.on_clicked(toggle_all)

    cluster_visible = {}

    button_x = 0.8
    button_width = 0.17
    button_height = 0.028
    gap = 0.004
    start_y = 0.95

    for i, cluster in enumerate(sorted(clusters_to_categories)):

        y = start_y - i * (button_height + gap)

        if y < 0.08:
            break

        cluster_ax = plt.axes(
            [button_x, y, button_width, button_height]
        )

        button = Button(cluster_ax, str(cluster))
        cluster_visible[cluster] = True

        def make_callback(cluster_name, axis=cluster_ax):

            def callback(event):

                state = not cluster_visible[cluster_name]
                cluster_visible[cluster_name] = state

                for category in clusters_to_categories[cluster_name]:
                    if category in line_by_category:
                        line_by_category[category].set_visible(state)

                axis.set_facecolor("0.85" if state else "0.65")

                update_legend()

            return callback

        button.on_clicked(make_callback(cluster))

    ax.set_title("Sales dynamics for categories")
    ax.set_xlabel("date_block_num")
    ax.set_ylabel("log(Number of sales + 1)")

    ax.grid(True)

    plt.show()






def plot_top_items(df, top_n=30):
    """
    Plot top items by total sales.
    """

    top_items = (
        df.groupby("item_name")["item_cnt_day"]
        .sum()
        .sort_values()
        .tail(top_n)
    )

    plt.figure(figsize=(10, 12))

    top_items.plot(
        kind="barh",
        color="lightgreen",
        edgecolor="gray",
    )

    plt.title(f"Top-{top_n} products based on sales")
    plt.xlabel("Number of units sold")
    plt.ylabel("Item name")

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()




def plot_correlation_matrix(df, numeric_cols, n_rows=8):
    """
    Plot a correlation matrix for selected numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list[str]
        List of numeric columns.
    n_rows : int, default=8
        Number of rows from the correlation matrix to display.
    """

    corr = df[numeric_cols].corr()
    corr_subset = corr.iloc[:n_rows, :]

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        corr_subset,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )

    plt.title(f"Correlation matrix (first {n_rows} rows)")
    plt.tight_layout()
    plt.show()