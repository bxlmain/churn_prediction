import pandas as pd
import numpy as np


def create_snapshot_dates(config: dict) -> pd.DatetimeIndex:
    return pd.date_range(
        start=config["snapshots"]["start_date"],
        end=config["snapshots"]["end_date"],
        freq=config["snapshots"]["freq"],
    )


def build_customer_snapshot_base(
    customers: pd.DataFrame, snapshot_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    base = customers[
        [
            "customer_id",
            "registration_date",
            "city",
            "age",
            "gender",
            "preferred_payment",
        ]
    ].copy()

    snapshots = pd.DataFrame({"snapshot_date": snapshot_dates})
    base["key"] = 1
    snapshots["key"] = 1

    result = base.merge(snapshots, on="key").drop(columns="key")

    result = result[result["registration_date"] <= result["snapshot_date"]].copy()

    result["customer_lifetime_days"] = (
        result["snapshot_date"] - result["registration_date"]
    ).dt.days

    return result


def add_order_features(base: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for snapshot_date in base["snapshot_date"].unique():
        snapshot_date = pd.Timestamp(snapshot_date)

        base_part = base[base["snapshot_date"] == snapshot_date].copy()
        orders_hist = orders[(orders["order_date"] < snapshot_date)].copy()
        delivered_hist = orders_hist[orders_hist["status"] == "delivered"].copy()

        agg_all = (
            orders_hist.groupby("customer_id")
            .agg(
                orders_total=("order_id", "count"),
                orders_amount_total=("amount", "sum"),
                orders_amount_mean=("amount", "mean"),
                orders_quantity_total=("quantity", "sum"),
                orders_cancelled_total=("status", lambda x: (x == "cancelled").sum()),
            )
            .reset_index()
        )

        agg_delivered = (
            delivered_hist.groupby("customer_id")
            .agg(
                delivered_orders_total=("order_id", "count"),
                delivered_amount_total=("amount", "sum"),
                delivered_amount_mean=("amount", "mean"),
                last_delivered_order_date=("order_date", "max"),
            )
            .reset_index()
        )

        recent_30 = (
            delivered_hist[
                delivered_hist["order_date"] >= snapshot_date - pd.Timedelta(days=30)
            ]
            .groupby("customer_id")
            .agg(
                delivered_orders_30d=("order_id", "count"),
                delivered_amount_30d=("amount", "sum"),
            )
            .reset_index()
        )

        recent_90 = (
            delivered_hist[
                delivered_hist["order_date"] >= snapshot_date - pd.Timedelta(days=90)
            ]
            .groupby("customer_id")
            .agg(
                delivered_orders_90d=("order_id", "count"),
                delivered_amount_90d=("amount", "sum"),
            )
            .reset_index()
        )

        base_part = base_part.merge(agg_all, on="customer_id", how="left")
        base_part = base_part.merge(agg_delivered, on="customer_id", how="left")
        base_part = base_part.merge(recent_30, on="customer_id", how="left")
        base_part = base_part.merge(recent_90, on="customer_id", how="left")

        base_part["days_since_last_order"] = (
            snapshot_date - base_part["last_delivered_order_date"]
        ).dt.days

        base_part = base_part.drop(columns=["last_delivered_order_date"])

        rows.append(base_part)

    return pd.concat(rows, ignore_index=True)


def add_visit_features(base: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for snapshot_date in base["snapshot_date"].unique():
        snapshot_date = pd.Timestamp(snapshot_date)

        base_part = base[base["snapshot_date"] == snapshot_date].copy()

        visits_hist = visits[visits["visit_time"] < snapshot_date].copy()

        agg_all = (
            visits_hist.groupby("customer_id")
            .agg(
                visits_total=("visit_id", "count"),
                pages_viewed_total=("pages_viewed", "sum"),
                pages_viewed_mean=("pages_viewed", "mean"),
                cart_adds_total=("cart_adds", "sum"),
                checkout_initiated_total=("checkout_initiated", "sum"),
                session_duration_mean=("session_duration_sec", "mean"),
                last_visit_time=("visit_time", "max"),
            )
            .reset_index()
        )

        recent_30 = (
            visits_hist[
                visits_hist["visit_time"] >= snapshot_date - pd.Timedelta(days=30)
            ]
            .groupby("customer_id")
            .agg(
                visits_30d=("visit_id", "count"),
                cart_adds_30d=("cart_adds", "sum"),
                checkout_initiated_30d=("checkout_initiated", "sum"),
            )
            .reset_index()
        )

        recent_90 = (
            visits_hist[
                visits_hist["visit_time"] >= snapshot_date - pd.Timedelta(days=90)
            ]
            .groupby("customer_id")
            .agg(
                visits_90d=("visit_id", "count"),
                cart_adds_90d=("cart_adds", "sum"),
                checkout_initiated_90d=("checkout_initiated", "sum"),
            )
            .reset_index()
        )

        base_part = base_part.merge(agg_all, on="customer_id", how="left")
        base_part = base_part.merge(recent_30, on="customer_id", how="left")
        base_part = base_part.merge(recent_90, on="customer_id", how="left")

        base_part["days_since_last_visit"] = (
            snapshot_date - base_part["last_visit_time"]
        ).dt.days

        base_part = base_part.drop(columns=["last_visit_time"])

        rows.append(base_part)

    return pd.concat(rows, ignore_index=True)


def add_support_features(
    base: pd.DataFrame, support_tickets: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    support_tickets = support_tickets.copy()
    support_tickets["resolution_days"] = (
        support_tickets["closed_date"] - support_tickets["created_date"]
    ).dt.days

    for snapshot_date in base["snapshot_date"].unique():
        snapshot_date = pd.Timestamp(snapshot_date)

        base_part = base[base["snapshot_date"] == snapshot_date].copy()

        tickets_hist = support_tickets[
            support_tickets["created_date"] < snapshot_date
        ].copy()

        agg_all = (
            tickets_hist.groupby("customer_id")
            .agg(
                support_tickets_total=("ticket_id", "count"),
                support_rating_mean=("rating", "mean"),
                support_resolution_days_mean=("resolution_days", "mean"),
                last_ticket_date=("created_date", "max"),
            )
            .reset_index()
        )

        recent_90 = (
            tickets_hist[
                tickets_hist["created_date"] >= snapshot_date - pd.Timedelta(days=90)
            ]
            .groupby("customer_id")
            .agg(
                support_tickets_90d=("ticket_id", "count"),
            )
            .reset_index()
        )

        base_part = base_part.merge(agg_all, on="customer_id", how="left")
        base_part = base_part.merge(recent_90, on="customer_id", how="left")

        base_part["days_since_last_ticket"] = (
            snapshot_date - base_part["last_ticket_date"]
        ).dt.days

        base_part = base_part.drop(columns=["last_ticket_date"])

        rows.append(base_part)

    return pd.concat(rows, ignore_index=True)


def add_target(
    base: pd.DataFrame, orders: pd.DataFrame, horizon_days: int
) -> pd.DataFrame:
    rows = []

    delivered_orders = orders[orders["status"] == "delivered"].copy()

    for snapshot_date in base["snapshot_date"].unique():
        snapshot_date = pd.Timestamp(snapshot_date)
        horizon_date = snapshot_date + pd.Timedelta(days=horizon_days)

        base_part = base[base["snapshot_date"] == snapshot_date].copy()

        future_orders = delivered_orders[
            (delivered_orders["order_date"] > snapshot_date)
            & (delivered_orders["order_date"] <= horizon_date)
        ]

        active_customers = future_orders["customer_id"].unique()

        base_part["churn_flag"] = np.where(
            base_part["customer_id"].isin(active_customers), 0, 1
        )

        rows.append(base_part)

    return pd.concat(rows, ignore_index=True)


def build_features(data: dict) -> pd.DataFrame:
    customers = data["customers"]
    orders = data["orders"]
    visits = data["visits"]
    support_tickets = data["support_tickets"]
    config = data["config"]

    print("Building snapshot dataset...")

    snapshot_dates = create_snapshot_dates(config)

    dataset = build_customer_snapshot_base(customers, snapshot_dates)
    dataset = add_order_features(dataset, orders)
    dataset = add_visit_features(dataset, visits)
    dataset = add_support_features(dataset, support_tickets)
    dataset = add_target(dataset, orders, config["target"]["horizon_days"])

    numeric_columns = dataset.select_dtypes(include=["number"]).columns
    dataset[numeric_columns] = dataset[numeric_columns].fillna(0)

    categorical_columns = ["city", "gender", "preferred_payment"]
    for col in categorical_columns:
        dataset[col] = dataset[col].fillna("unknown")

    dataset = dataset.sort_values(["snapshot_date", "customer_id"]).reset_index(
        drop=True
    )

    print(f"Final ML dataset shape: {dataset.shape}")
    print("Target distribution:")
    print(dataset["churn_flag"].value_counts(normalize=True))

    return dataset
