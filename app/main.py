import os
from datetime import date

from dotenv import load_dotenv

# Must run before the app.* imports below, since those modules read
# environment variables (GROQ_API_KEY, DATABASE_URL, JWT_SECRET_KEY) at import time.
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import get_db, User, Rewrite, DailyUsage
from app.schemas import (
    SignupRequest, LoginRequest, TokenResponse,
    RewriteRequest, RewriteResponse, HistoryItem,
)
from app.auth import hash_password, verify_password, create_access_token, get_current_user
from app.ai import rewrite_text, get_available_tones

FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", "20"))

app = FastAPI(title="ToneShift API", version="1.0.0")

# Comma-separated list of allowed web origins, e.g.
#   ALLOWED_ORIGINS=https://toneshift.app,https://www.toneshift.app
# Defaults to "*" (fine for native mobile apps, which don't use CORS; set it
# explicitly in production if you also have a web frontend).
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "ToneShift API", "status": "ok", "docs": "/docs"}


# ---------- Auth ----------

@app.post("/auth/signup", response_model=TokenResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


# ---------- Tones ----------

@app.get("/tones")
def list_tones():
    return {"tones": get_available_tones()}


# ---------- Rewrite (core feature) ----------

def _check_and_increment_quota(user: User, db: Session):
    """Free users get FREE_DAILY_LIMIT rewrites/day. Pro users are unlimited."""
    if user.is_pro:
        return

    today = date.today()
    usage = (
        db.query(DailyUsage)
        .filter(DailyUsage.user_id == user.id, DailyUsage.usage_date == today)
        .first()
    )
    if usage is None:
        usage = DailyUsage(user_id=user.id, usage_date=today, count=0)
        db.add(usage)

    if usage.count >= FREE_DAILY_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Free daily limit of {FREE_DAILY_LIMIT} rewrites reached. Upgrade to Pro for unlimited access.",
        )

    usage.count += 1
    db.commit()


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(
    payload: RewriteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_and_increment_quota(current_user, db)

    # Cache check: same user + same text + same tone -> reuse stored result,
    # skip calling the AI model again (saves cost and is instant).
    cached = (
        db.query(Rewrite)
        .filter(
            Rewrite.user_id == current_user.id,
            Rewrite.original_text == payload.text,
            Rewrite.tone == payload.tone,
            Rewrite.rewritten_text != "",  # ignore any bad empty rows saved earlier
        )
        .first()
    )
    if cached:
        return RewriteResponse(rewritten_text=cached.rewritten_text, tone=payload.tone, cached=True)

    try:
        result_text = rewrite_text(payload.text, payload.tone)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Surface the real reason (bad API key, wrong model name, Groq outage, etc.)
        # instead of a bare "Internal Server Error" with no explanation.
        raise HTTPException(
            status_code=502,
            detail=f"AI provider call failed: {type(e).__name__}: {e}",
        )

    record = Rewrite(
        user_id=current_user.id,
        original_text=payload.text,
        tone=payload.tone,
        rewritten_text=result_text,
    )
    db.add(record)
    db.commit()

    return RewriteResponse(rewritten_text=result_text, tone=payload.tone, cached=False)


# ---------- History ----------

@app.get("/history", response_model=list[HistoryItem])
def history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 50,
):
    rows = (
        db.query(Rewrite)
        .filter(Rewrite.user_id == current_user.id)
        .order_by(Rewrite.created_at.desc())
        .limit(limit)
        .all()
    )
    return rows


# ---------- Health check ----------

@app.get("/health")
def health():
    return {"status": "ok"}
