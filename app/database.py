import os
from datetime import datetime, date

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Date, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Copy .env.example to .env and paste your Supabase "
        "connection string (Supabase project -> Settings -> Database -> Connection string -> URI)."
    )

# Some tools hand out "postgres://" URLs; SQLAlchemy 2.x needs "postgresql://".
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# pool_pre_ping avoids "server closed the connection" errors after Supabase's
# connection pooler drops an idle connection.
#
# On Vercel (serverless) every instance is short-lived and there can be many of
# them at once, so we don't keep a local connection pool (NullPool) and let
# Supabase's Transaction pooler (port 6543) do the pooling instead.
_engine_kwargs = {"pool_pre_ping": True}
if os.getenv("VERCEL"):
    _engine_kwargs["poolclass"] = NullPool

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_pro = Column(Integer, default=0)  # 0 = free, 1 = pro/paid
    created_at = Column(DateTime, default=datetime.utcnow)

    rewrites = relationship("Rewrite", back_populates="user")
    usage = relationship("DailyUsage", back_populates="user")


class Rewrite(Base):
    __tablename__ = "rewrites"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_text = Column(Text, nullable=False)
    tone = Column(String, nullable=False)
    rewritten_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="rewrites")


class DailyUsage(Base):
    """Tracks how many free rewrites a user has used on a given day."""
    __tablename__ = "daily_usage"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    usage_date = Column(Date, default=date.today)
    count = Column(Integer, default=0)

    user = relationship("User", back_populates="usage")


_db_initialized = False


def init_db():
    """Create tables once per process (safe to call repeatedly)."""
    global _db_initialized
    if _db_initialized:
        return
    Base.metadata.create_all(bind=engine)
    _db_initialized = True


def get_db():
    # Serverless platforms don't reliably run FastAPI startup events, so make
    # sure the tables exist lazily on the first request of each instance.
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
