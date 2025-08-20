import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
ENGINE = None  # will be set by setup_database()

class StockData(Base):
    __tablename__ = "stock_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("ticker", "timestamp", name="uix_ticker_timestamp"),
    )

# You can call setup_database() to create the DB and get a Session factory.
def setup_database(db_path: str | None = None):
    """
    Create engine, ensure tables exist and return a Session factory.
    Sets module-level ENGINE for direct core executions (used for upserts).
    """
    global ENGINE
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_dir = os.path.join(project_root, "sql_db")
    os.makedirs(db_dir, exist_ok=True)

    if db_path is None:
        db_file = os.path.join(db_dir, "realtime_anomaly.db")
        db_url = f"sqlite:///{db_file}"
    else:
        db_url = db_path

    ENGINE = create_engine(db_url, echo=False, future=True)
    Base.metadata.create_all(bind=ENGINE)
    SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, future=True)
    return SessionLocal

def get_engine():
    return ENGINE

# convenience alias that run_all.py can call to ensure DB is initialized
def init_db(db_path: str | None = None):
    return setup_database(db_path)