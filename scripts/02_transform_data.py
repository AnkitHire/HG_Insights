import pandas as pd
import pyodbc
import configparser
from pathlib import Path
from datetime import datetime
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_transform.log'),
        logging.StreamHandler()
    ]
)

def get_db_connection(config, db_type):
    """Get database connection string with validation"""
    required_keys = [f'{db_type}_server', f'{db_type}_database']
    if not all(key in config for key in required_keys):
        raise ValueError(f"Missing required config keys for {db_type} database")
    
    return (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={config[f'{db_type}_server']};"
        f"DATABASE={config[f'{db_type}_database']};"
        f"Trusted_Connection=yes;"
        f"Encrypt=no;"
    )

def validate_dataframe(df, required_columns):
    """Validate the dataframe structure and content"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    logging.info(f"Data validation passed with {len(df)} records")

def transform_data():
    try:
        # Set pandas options
        pd.set_option('future.no_silent_downcasting', True)
        pd.set_option('mode.chained_assignment', 'raise')
        
        # Load configuration
        config = configparser.ConfigParser()
        config_file = 'C:/Users/ankit.h/Pictures/Saved Pictures/personal/HG/config/config.ini'
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        config.read(config_file)
        logging.info("Configuration loaded successfully")

        if not config.has_section('databases'):
            raise ValueError("Missing [databases] section in config.ini")

        db_config = config['databases']
        
        # Connect to staging
        logging.info("Connecting to staging database")
        staging_conn_str = get_db_connection(db_config, 'staging')
        reporting_conn_str = get_db_connection(db_config, 'reporting')
        with pyodbc.connect(staging_conn_str) as conn:
            df = pd.read_sql("SELECT * FROM churn_staging", conn)
        
        # Validate input data
        required_columns = [
            'CustomerID','MonthlyCharges', 'TotalCharges', 'Age', 'Tenure',
            'ContractType', 'InternetService', 'TechSupport', 'Gender', 'Churn'
        ]
        validate_dataframe(df, required_columns)

        # 1. Handle missing values with logging
        logging.info("Handling missing values")
        numeric_cols = ['MonthlyCharges', 'TotalCharges', 'Age', 'Tenure','CustomerID']
        for col in numeric_cols:
            initial_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            
            # Ensure values are within SQL Server range
            if col in ['MonthlyCharges', 'TotalCharges']:
                df[col] = df[col].clip(lower=0, upper=999999.99)
            
            final_nulls = df[col].isna().sum()
            logging.info(f"Column {col}: {initial_nulls} nulls handled, {final_nulls} remain")

        categorical_cols = ['ContractType', 'InternetService', 'TechSupport', 'Gender', 'Churn']
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
            logging.info(f"Column {col}: Filled nulls with 'Unknown'")

        # 2. Standardize values with explicit type conversion
        logging.info("Standardizing values")
        contract_mapping = {
            'Mtm': 'Month-to-Month',
            'Annual': 'Yearly',
            '2Year': '2-Year'
        }
        df['ContractType'] = (
            df['ContractType']
            .astype(str)
            .str.strip()
            .str.title()
            .replace(contract_mapping)
        )
        
        internet_mapping = {
            'dsl': 'DSL',
            'fiber': 'Fiber Optic',
            'none': 'No Internet'
        }
        df['InternetService'] = (
            df['InternetService']
            .astype(str)
            .str.strip()
            .replace(internet_mapping)
        )
        
        # Convert boolean-like columns to integers (1/0)
        tech_support_mapping = {
            'yes': 1,
            'no': 0,
            'true': 1,
            'false': 0
        }
        df['TechSupport'] = (
            df['TechSupport']
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(tech_support_mapping)
            .astype(int)
        )
        
        churn_mapping = {
            'yes': 1,
            'no': 0
        }
        df['Churn'] = (
            df['Churn']
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(churn_mapping)
            .astype(int)
        )

        # 3. Calculate derived fields with validation
        logging.info("Calculating derived fields")
        df['Tenure'] = df['Tenure'].clip(lower=0)  # Ensure tenure is non-negative
        
        def calculate_customer_since(tenure_months):
            try:
                tenure_months = int(tenure_months)
                if tenure_months < 0:
                    return datetime.now().date()
                return (datetime.now() - pd.DateOffset(months=tenure_months)).date()
            except:
                return datetime.now().date()
        
        df['CustomerSince'] = df['Tenure'].apply(calculate_customer_since)
        
        df['RevenuePerMonth'] = (
            df['TotalCharges'] / np.where(df['Tenure'] > 0, df['Tenure'], 1)
        ).round(2)

        # ========== CREATE DIMENSIONAL MODEL ==========
        logging.info("Creating dimensional model")
        with pyodbc.connect(reporting_conn_str) as conn:
            cursor = conn.cursor()
            
            # Drop existing tables if they exist
            tables = [
                'FactCustomerChurn', 'DimCustomer', 'DimContract', 
                'DimInternetService', 'DimTechSupport', 'ReportChurnAnalysis'
            ]
            for table in tables:
                cursor.execute(f"IF OBJECT_ID('reporting.{table}') IS NOT NULL DROP TABLE reporting.{table}")
            conn.commit()
            logging.info("Dropped existing tables")

            # Create dimension tables with explicit data types
            dimension_tables = [
                ("""CREATE TABLE reporting.DimCustomer (
                    CustomerKey INT IDENTITY(1,1) PRIMARY KEY,
                    CustomerID VARCHAR(20),
                    Age INT,
                    Gender VARCHAR(20),
                    CustomerSince DATE
                )""", "DimCustomer"),
                
                ("""CREATE TABLE reporting.DimContract (
                    ContractKey INT IDENTITY(1,1) PRIMARY KEY,
                    ContractType NVARCHAR(50)
                )""", "DimContract"),
                
                ("""CREATE TABLE reporting.DimInternetService (
                    InternetServiceKey INT IDENTITY(1,1) PRIMARY KEY,
                    InternetService NVARCHAR(50)
                )""", "DimInternetService"),
                
                ("""CREATE TABLE reporting.DimTechSupport (
                    TechSupportKey INT IDENTITY(1,1) PRIMARY KEY,
                    TechSupport TINYINT
                )""", "DimTechSupport")
            ]
            
            for create_stmt, table_name in dimension_tables:
                cursor.execute(create_stmt)
                logging.info(f"Created table {table_name}")
            
            # Create fact table with constraints
            cursor.execute("""
                CREATE TABLE reporting.FactCustomerChurn (
                    FactKey INT IDENTITY(1,1) PRIMARY KEY,
                    CustomerKey INT NOT NULL FOREIGN KEY REFERENCES reporting.DimCustomer(CustomerKey),
                    ContractKey INT NOT NULL FOREIGN KEY REFERENCES reporting.DimContract(ContractKey),
                    InternetServiceKey INT NOT NULL FOREIGN KEY REFERENCES reporting.DimInternetService(InternetServiceKey),
                    TechSupportKey INT NOT NULL FOREIGN KEY REFERENCES reporting.DimTechSupport(TechSupportKey),
                    Tenure INT NOT NULL CHECK (Tenure >= 0),
                    MonthlyCharges DECIMAL(10,2) NOT NULL CHECK (MonthlyCharges >= 0),
                    TotalCharges DECIMAL(10,2) NOT NULL CHECK (TotalCharges >= 0),
                    RevenuePerMonth DECIMAL(10,2),
                    ChurnStatus TINYINT NOT NULL CHECK (ChurnStatus IN (0,1)),
                    LoadDate DATE DEFAULT GETDATE()
                )
            """)
            logging.info("Created FactCustomerChurn table")
            
            # Create reporting table
            cursor.execute("""
                CREATE TABLE reporting.ReportChurnAnalysis (
                    ReportKey INT IDENTITY(1,1) PRIMARY KEY,
                    ContractType NVARCHAR(50),
                    InternetService NVARCHAR(50),
                    HasTechSupport TINYINT,
                    CustomerCount INT,
                    AvgTenure DECIMAL(10,2),
                    AvgMonthlyCharges DECIMAL(10,2),
                    ChurnRate DECIMAL(5,4),
                    LoadDate DATE DEFAULT GETDATE()
                )
            """)
            logging.info("Created ReportChurnAnalysis table")
            
            conn.commit()

            # ========== LOAD DATA ==========
            logging.info("Loading dimension tables")
            dim_data = {
                'Customer': df[['CustomerID', 'Age', 'Gender', 'CustomerSince']].drop_duplicates(),
                'Contract': df[['ContractType']].drop_duplicates(),
                'InternetService': df[['InternetService']].drop_duplicates(),
                'TechSupport': df[['TechSupport']].drop_duplicates()
            }
            
            key_mappings = {}
            batch_size = 1000  # Process in batches for large datasets
            
            for dim_name, dim_df in dim_data.items():
                table_name = f"Dim{dim_name}"
                cols = [col for col in dim_df.columns if col != 'original_index']
                
                # Insert data in batches
                for i in range(0, len(dim_df), batch_size):
                    batch = dim_df.iloc[i:i+batch_size]
                    data_tuples = [tuple(row) for row in batch.values.tolist()]
                    
                    try:
                        cursor.fast_executemany = True
                        insert_sql = f"INSERT INTO reporting.{table_name} ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
                        cursor.executemany(insert_sql, data_tuples)
                        conn.commit()
                        logging.info(f"Loaded batch {i//batch_size + 1} for {table_name}")
                    except Exception as e:
                        conn.rollback()
                        logging.error(f"Error loading batch {i//batch_size + 1} for {table_name}: {str(e)}")
                        raise
                
                # Get mapping of original values to generated keys
                cursor.execute(f"SELECT {cols[0]}, {dim_name}Key FROM reporting.{table_name}")
                key_mappings[dim_name] = {row[0]: row[1] for row in cursor.fetchall()}
                logging.info(f"Mapped {len(key_mappings[dim_name])} keys for {table_name}")

            # Verify all foreign keys exist
            missing_keys = False
            for col, dim_name in [('CustomerID', 'Customer'), 
                                ('ContractType', 'Contract'),
                                ('InternetService', 'InternetService'),
                                ('TechSupport', 'TechSupport')]:
                
                missing_mask = ~df[col].isin(key_mappings.get(dim_name, {}))
                missing_count = missing_mask.sum()
                
                if missing_count > 0:
                    logging.error(f"Critical: {missing_count} records still missing {dim_name} references")
                    # Log sample problematic records
                    problem_samples = df.loc[missing_mask, [col]].head(5).to_dict('records')
                    logging.error(f"Problem samples: {problem_samples}")
                    missing_keys = True

            if missing_keys:
                raise ValueError("Critical foreign key references missing after remediation")
                    # Prepare fact data with validation
            fact_data = pd.DataFrame({
                'CustomerKey': df['CustomerID'].map(key_mappings['Customer']),
                'ContractKey': df['ContractType'].map(key_mappings['Contract']),
                'InternetServiceKey': df['InternetService'].map(key_mappings['InternetService']),
                'TechSupportKey': df['TechSupport'].map(key_mappings['TechSupport']),
                'Tenure': df['Tenure'].astype(int),
                'MonthlyCharges': df['MonthlyCharges'].astype(float),
                'TotalCharges': df['TotalCharges'].astype(float),
                'RevenuePerMonth': df['RevenuePerMonth'].astype(float),
                'ChurnStatus': df['Churn'].astype(int)
            })
            logging.info(fact_data.head())
            logging.info(fact_data.isnull().sum())
            # Insert fact data in batches
            logging.info("Loading fact table")
            for i in range(0, len(fact_data), batch_size):
                batch = fact_data.iloc[i:i+batch_size]
                insert_data = [
                    (
                        int(row.CustomerKey),
                        int(row.ContractKey),
                        int(row.InternetServiceKey),
                        int(row.TechSupportKey),
                        int(row.Tenure),
                        float(row.MonthlyCharges),
                        float(row.TotalCharges),
                        float(row.RevenuePerMonth),
                        int(row.ChurnStatus)
                    )
                    for row in batch.itertuples()
                ]
                
                try:
                    cursor.fast_executemany = True
                    cursor.executemany("""
                        INSERT INTO reporting.FactCustomerChurn (
                            CustomerKey, ContractKey, InternetServiceKey, TechSupportKey,
                            Tenure, MonthlyCharges, TotalCharges, RevenuePerMonth, ChurnStatus
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, insert_data)
                    conn.commit()
                    logging.info(f"Loaded fact batch {i//batch_size + 1}")
                except Exception as e:
                    conn.rollback()
                    logging.error(f"Error loading fact batch {i//batch_size + 1}: {str(e)}")
                    raise

            # Generate and load report data
            logging.info("Generating report data")
            report_data = df.groupby(['ContractType', 'InternetService', 'TechSupport']).agg(
                CustomerCount=('CustomerID', 'count'),
                AvgTenure=('Tenure', 'mean'),
                AvgMonthlyCharges=('MonthlyCharges', 'mean'),
                ChurnRate=('Churn', 'mean')
            ).reset_index()

            # Insert report data
            cursor.fast_executemany = True
            report_insert_data = [
                (
                    str(row.ContractType),
                    str(row.InternetService),
                    int(row.TechSupport),
                    int(row.CustomerCount),
                    float(row.AvgTenure),
                    float(row.AvgMonthlyCharges),
                    float(row.ChurnRate)
                )
                for row in report_data.itertuples()
            ]
            
            try:
                cursor.executemany("""
                    INSERT INTO reporting.ReportChurnAnalysis (
                        ContractType, InternetService, HasTechSupport,
                        CustomerCount, AvgTenure, AvgMonthlyCharges, ChurnRate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, report_insert_data)
                conn.commit()
                logging.info("Loaded report data")
            except Exception as e:
                conn.rollback()
                logging.error(f"Error loading report data: {str(e)}")
                raise

        logging.info("ETL process completed successfully with dimensional model")
        return True

    except Exception as e:
        logging.error(f"ETL process failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    transform_data()