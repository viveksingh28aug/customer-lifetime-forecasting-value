"""Database Management Layer for CLV Engine.

Provides an abstraction layer supporting DuckDB (default), SQLite (fallback),
and optional PostgreSQL with parameterized queries, schema management, and transactional safety.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import duckdb

from config.settings import (
    DB_BACKEND,
    DB_DUCKDB_PATH,
    DB_SQLITE_PATH,
    DB_PG_HOST,
    DB_PG_PORT,
    DB_PG_NAME,
    DB_PG_USER,
    DB_PG_PASSWORD,
    DATABASE_DIR,
)
from config.logging_config import get_logger

logger = get_logger("database.db_manager")

class DatabaseManager:
    """Manages analytical database connections, schema initialization, and queries."""

    def __init__(self, backend: Optional[str] = None, db_path: Optional[str] = None):
        self.backend = (backend or DB_BACKEND).lower()
        self.db_path = db_path
        self._conn = None
        self._init_connection()

    def _init_connection(self):
        """Initializes connection based on selected backend."""
        try:
            if self.backend == "duckdb":
                path = str(self.db_path or DB_DUCKDB_PATH)
                logger.info(f"Connecting to DuckDB database at: {path}")
                self._conn = duckdb.connect(database=path, read_only=False)
            elif self.backend == "sqlite":
                path = str(self.db_path or DB_SQLITE_PATH)
                logger.info(f"Connecting to SQLite database at: {path}")
                self._conn = sqlite3.connect(path, check_same_thread=False)
            elif self.backend == "postgres":
                logger.info("Connecting to PostgreSQL backend...")
                import psycopg2
                self._conn = psycopg2.connect(
                    host=DB_PG_HOST,
                    port=DB_PG_PORT,
                    dbname=DB_PG_NAME,
                    user=DB_PG_USER,
                    password=DB_PG_PASSWORD,
                )
            else:
                logger.warning(f"Unknown database backend '{self.backend}'. Falling back to DuckDB.")
                self.backend = "duckdb"
                self._conn = duckdb.connect(database=str(DB_DUCKDB_PATH), read_only=False)
        except Exception as e:
            logger.error(f"Failed to connect to backend '{self.backend}': {e}. Falling back to SQLite in-memory.")
            self.backend = "sqlite"
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

    def init_schema(self, schema_file_path: Optional[str] = None):
        """Creates required analytical tables if they do not exist."""
        if schema_file_path is None:
            schema_file_path = DATABASE_DIR / "schema.sql"

        try:
            with open(schema_file_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            # Execute statements sequentially
            statements = [stmt.strip() for stmt in schema_sql.split(";") if stmt.strip()]
            for stmt in statements:
                self.execute_statement(stmt)
            logger.info("Database schema initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            raise

    def execute_statement(self, sql: str, params: Optional[Union[tuple, list]] = None):
        """Executes a single DDL or DML statement with optional parameters."""
        if self._conn is None:
            self._init_connection()

        params = params or ()
        try:
            if self.backend == "duckdb":
                if params:
                    self._conn.execute(sql, params)
                else:
                    self._conn.execute(sql)
            elif self.backend == "sqlite":
                cursor = self._conn.cursor()
                cursor.execute(sql, params)
                self._conn.commit()
            elif self.backend == "postgres":
                with self._conn.cursor() as cursor:
                    cursor.execute(sql, params)
                self._conn.commit()
        except Exception as e:
            logger.error(f"Database execution failed for query: {sql[:100]}... Error: {e}")
            raise

    def query(self, sql: str, params: Optional[Union[tuple, list]] = None) -> pd.DataFrame:
        """Executes a SELECT query and returns the results as a Pandas DataFrame."""
        if self._conn is None:
            self._init_connection()

        params = params or ()
        try:
            if self.backend == "duckdb":
                if params:
                    return self._conn.execute(sql, params).df()
                return self._conn.execute(sql).df()
            elif self.backend == "sqlite":
                return pd.read_sql_query(sql, self._conn, params=params if params else None)
            elif self.backend == "postgres":
                return pd.read_sql_query(sql, self._conn, params=params if params else None)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Query execution failed: {sql[:100]}... Error: {e}")
            raise

    def save_dataframe(self, table_name: str, df: pd.DataFrame, if_exists: str = "replace"):
        """Saves a Pandas DataFrame into the specified table."""
        if df.empty:
            logger.warning(f"Skipping save: DataFrame for table '{table_name}' is empty.")
            return

        if self._conn is None:
            self._init_connection()

        try:
            if self.backend == "duckdb":
                # DuckDB native registration and write
                self._conn.register("df_temp_view", df)
                if if_exists == "replace":
                    self._conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_temp_view")
                elif if_exists == "append":
                    self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM df_temp_view")
                self._conn.unregister("df_temp_view")
            elif self.backend == "sqlite":
                df.to_sql(table_name, self._conn, if_exists=if_exists, index=False)
            elif self.backend == "postgres":
                df.to_sql(table_name, self._conn, if_exists=if_exists, index=False)
            logger.info(f"Successfully saved {len(df)} rows into table '{table_name}' ({if_exists}).")
        except Exception as e:
            logger.error(f"Failed to save DataFrame into table '{table_name}': {e}")
            raise

    def close(self):
        """Closes the active database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info(f"Closed {self.backend} connection.")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self._conn = None
