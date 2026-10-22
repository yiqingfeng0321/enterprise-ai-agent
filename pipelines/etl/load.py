from __future__ import annotations

import pandas as pd

from app.repositories.database import Database


def load_feature_table(frame: pd.DataFrame, database: Database, table_name: str) -> int:
    with database.connect() as connection:
        frame.to_sql(table_name, connection, if_exists="replace", index=False)
        connection.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}(customer_id)"
        )
    return int(len(frame))
