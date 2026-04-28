import pandas as pd
import yaml


def read_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(config_path: str = "configs/config.yaml") -> dict:
    config = read_config(config_path)

    customers = pd.read_csv(config["data"]["customers_path"])
    orders = pd.read_csv(config["data"]["orders_path"])
    visits = pd.read_csv(config["data"]["visits_path"])
    support_tickets = pd.read_csv(config["data"]["support_tickets_path"])

    customers["registration_date"] = pd.to_datetime(customers["registration_date"])

    orders["order_date"] = pd.to_datetime(orders["order_date"])

    visits["visit_time"] = pd.to_datetime(visits["visit_time"])

    support_tickets["created_date"] = pd.to_datetime(support_tickets["created_date"])
    support_tickets["closed_date"] = pd.to_datetime(support_tickets["closed_date"])

    print(f"Customers: {customers.shape}")
    print(f"Orders: {orders.shape}")
    print(f"Visits: {visits.shape}")
    print(f"Support tickets: {support_tickets.shape}")

    return {
        "customers": customers,
        "orders": orders,
        "visits": visits,
        "support_tickets": support_tickets,
        "config": config,
    }
