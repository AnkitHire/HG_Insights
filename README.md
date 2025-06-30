# HG_Insights

# Telecom Customer Churn ELT Pipeline (SQL Server Version)

This project implements an ELT pipeline for processing telecom customer churn data using SQL Server.

## Prerequisites

- Python 3.8+
- SQL Server (2016 or later)
- ODBC Driver 17 for SQL Server


## Setup Instructions

1. **SQL Server Setup**:
   - Create two databases: `StagingDB` and `ReportingDB`
   - Ensure you have login credentials with appropriate permissions

2. **Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
