import mysql.connector
import pandas as pd
from typing import Dict


def fetch_sql_sleep_data(db_config: Dict[str, str]) -> pd.DataFrame:
    try:
        connection = mysql.connector.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
        )

        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sleep_data")
        data = cursor.fetchall()

        return pd.DataFrame(data)

    except mysql.connector.Error as error:
        raise RuntimeError(f"Error while trying to access the database: {error}")

    finally:
        if "connection" in locals() and connection.is_connected():
            cursor.close()
            connection.close()
