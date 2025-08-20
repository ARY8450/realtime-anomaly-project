from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import os

Base = declarative_base()

class StockData(Base):
    __tablename__ = 'stock_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

def setup_database(db_path=None):
    """
    Create engine, ensure tables exist and return a Session factory (SessionLocal).
    Default DB file is stored under realtime_anomaly_project/sql_db/realtime_anomaly.db
    """
    # default sqlite file inside project under sql_db/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_dir = os.path.join(project_root, "sql_db")
    os.makedirs(db_dir, exist_ok=True)

    if db_path is None:
        db_file = os.path.join(db_dir, "realtime_anomaly.db")
        db_url = f"sqlite:///{db_file}"
    else:
        db_url = db_path

    engine = create_engine(db_url, echo=False, future=True)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
