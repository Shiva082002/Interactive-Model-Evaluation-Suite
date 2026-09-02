"""Synthetic analytics database for NL2SQL practice."""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

try:
    from .settings import DATABASE_PATH, DATA_DIR
except ImportError:
    from settings import DATABASE_PATH, DATA_DIR

SCHEMA = """CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, email TEXT, country TEXT, signup_date TEXT, plan TEXT, monthly_spend REAL, churned INTEGER, support_tickets INTEGER, last_active_date TEXT)"""


def initialize_database(rows: int = 20_000, force: bool = False) -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATABASE_PATH.exists() and not force:
        try:
            return database_summary()
        except sqlite3.OperationalError:
            pass
    rng = random.Random(42)
    names = ["Asha", "Luis", "Mina", "Noah", "Sofia", "Owen", "Priya", "Elena"]
    countries = ["US", "UK", "IN", "CA", "DE", "AU"]
    plans = ["starter", "pro", "business", "enterprise"]
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute("DROP TABLE IF EXISTS customers")
        connection.execute(SCHEMA)
        records = []
        for customer_id in range(1, rows + 1):
            plan = rng.choices(plans, weights=[45, 35, 15, 5])[0]
            price = {"starter": 19, "pro": 59, "business": 199, "enterprise": 799}[plan]
            records.append((customer_id, f"{rng.choice(names)} {customer_id}", f"customer{customer_id}@example.com", rng.choice(countries), f"202{rng.randrange(0, 6)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 28):02d}", plan, round(price * rng.uniform(.8, 1.25), 2), int(rng.random() < .12), rng.randrange(0, 25), "2026-08-01"))
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        connection.execute("CREATE INDEX idx_customers_plan ON customers(plan)")
        connection.execute("CREATE INDEX idx_customers_country ON customers(country)")
        connection.commit()
    finally:
        connection.close()
    return database_summary()


def database_summary() -> dict[str, object]:
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        count = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    finally:
        connection.close()
    return {"path": str(DATABASE_PATH), "table": "customers", "rows": count}


def schema_description() -> str:
    return SCHEMA
