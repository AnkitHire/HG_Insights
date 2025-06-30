import pandas as pd
import pyodbc
import configparser

config = configparser.ConfigParser()
config.read('C:/Users/ankit.h/Pictures/Saved Pictures/personal/HG/config/config.ini')

def load_to_staging():
    # Create connection string
    conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={config['databases']['staging_server']};"
            f"DATABASE={config['databases']['staging_database']};"
            f"Trusted_Connection=yes;"  # This enables Windows Authentication
            f"Encrypt=no;"  # Disable encryption for local connections
            f"Connection Timeout=30;"
        )
    # Read CSV
    df = pd.read_csv("C:/Users/ankit.h/Downloads/archive (3)/customer_churn_data.csv")

    # Convert numeric columns
    numeric_cols = ['CustomerID', 'Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert string columns
    string_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport', 'Churn']
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
    
    # Connect to SQL Server and load data
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='churn_staging' AND xtype='U')
        CREATE TABLE churn_staging (
            CustomerID INT,
            Age INT,
            Gender NVARCHAR(50),
            Tenure INT,
            MonthlyCharges DECIMAL(10,2),
            ContractType NVARCHAR(50),
            InternetService NVARCHAR(50),
            TotalCharges DECIMAL(10,2),
            TechSupport NVARCHAR(50),
            Churn NVARCHAR(50)
        )
        """)
        
        # Truncate table
        cursor.execute("TRUNCATE TABLE churn_staging")
        conn.commit()
        
        # Insert data
        for index, row in df.iterrows():
            cursor.execute("""
            INSERT INTO churn_staging VALUES (?,?,?,?,?,?,?,?,?,?)
            """, *row)
        
        conn.commit()

if __name__ == "__main__":
    load_to_staging()