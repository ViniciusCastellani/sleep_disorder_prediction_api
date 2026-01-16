import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv('Sleep_Health_Massive_Dataset.csv')
df = df.fillna('None')

try:
    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    
    if connection.is_connected():
        cursor = connection.cursor()
        print("Connected to database. Starting insertion...")

        sql_command = """
        INSERT INTO sleep_data (
            person_id, gender, age, occupation, sleep_duration, 
            quality_of_sleep, physical_activity_level, stress_level, 
            bmi_category, blood_pressure, heart_rate, daily_steps, sleep_disorder
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = [tuple(x) for x in df.values]

        cursor.executemany(sql_command, values)
        
        connection.commit()
        print(f"Success! {cursor.rowcount} rows inserted into sleep_data table.")

except mysql.connector.Error as error:
    print(f"Error connecting or inserting: {error}")

finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("Connection closed.")