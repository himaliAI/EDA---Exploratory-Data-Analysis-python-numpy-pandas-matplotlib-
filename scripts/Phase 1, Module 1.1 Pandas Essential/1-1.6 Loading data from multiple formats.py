import pandas as pd
import numpy as np

# load data from CSV
df_csv = pd.read_csv('d:\portfolio.csv')
#print(df_csv.columns.tolist())
print(df_csv.shape)

# Load Excel file
df_excel = pd.read_excel("D:\Anup's\Portfolio 2024.xlsm", sheet_name="Data")
#print(df_excel.head())
print(df_excel.shape)
print(df_excel.info())

'''
# Load json file
df_json = pd.read_json("filename.json")

# Load data from SQL
import sqlite3
# connect to SQLite database (or use SQLAlchemy for other DBs eg MySQL, PostgreSQL)
conn = sqlite3.connect("your_database.db")
# load table into DataFrame
df_sql = pd.read_sql("SELECT * FROM your_table", conn)
#
#
conn.close()
'''
