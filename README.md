# ToneShift Backend

FastAPI backend that rewrites text into a chosen tone (Gen Z, Formal, Corporate,
Sarcastic, Poetic, Casual) using Groq's free/fast LLM API. This is the API your
mobile app and Android keyboard will call.

## 1. Setup

```bash
cd toneshift-backend
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a free Groq API key

1. Go to https://console.groq.com/keys
2. Sign up (free) and create an API key
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Paste your key into `.env` as `GROQ_API_KEY=...`

## 3. Set up your Supabase database

1. Go to https://supabase.com and create a free project (takes ~2 minutes to provision)
2. In your project: **Settings -> Database -> Connection string -> URI**
3. Copy that connection string and paste it into `.env` as `DATABASE_URL=...`,
   replacing `[YOUR-PASSWORD]` with your actual database password
4. Use the **Transaction pooler** connection string (port `6543`) if you'll deploy
   this on a serverless host (Render, Railway, Vercel); use the **direct connection**
   (port `5432`) if it runs as one long-lived process — either works for local dev
5. Tables (`users`, `rewrites`, `daily_usage`) are created automatically the first
   time you run the server — you don't need to write any SQL yourself

You can open **Supabase -> Table Editor** any time to see your tables and rows
directly in the browser.

## 4. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`.
Interactive docs (auto-generated): `http://localhost:8000/docs`

## 5. Try it out

**Sign up:**
```bash
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'
```
This returns an `access_token` — copy it for the next steps.

**Rewrite text:**
```bash
curl -X POST http://localhost:8000/rewrite \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{"text": "bhai mujhe kal tak ye assignment submit karna hai, please jaldi reply karna", "tone": "corporate"}'
```

**List available tones:**
```bash
curl http://localhost:8000/tones
```

**View rewrite history:**
```bash
curl http://localhost:8000/history \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 6. How it fits together

```
Mobile App / Android Keyboard
        |
        |  POST /rewrite  { text, tone }
        v
   FastAPI Backend  ---->  checks cache (DB) ---->  found? return cached result
        |
        |  not cached
        v
   Groq API (openai/gpt-oss-120b)
        |
        v
   Rewritten text saved to DB + returned to app
```

## 7. Project structure

```
toneshift-backend/
├── app/
│   ├── main.py       # FastAPI routes (/auth, /rewrite, /tones, /history)
│   ├── ai.py          # Groq client + tone prompts + rewrite_text()
│   ├── auth.py        # Password hashing, JWT creation/validation
│   ├── database.py     # SQLAlchemy models (User, Rewrite, DailyUsage) + DB setup
│   └── schemas.py      # Pydantic request/response models
├── api/
│   └── index.py      # Vercel serverless entrypoint
├── vercel.json        # Vercel routing + function config
├── .python-version    # Python version used by Vercel
├── .vercelignore
├── .gitignore
├── requirements.txt
├── .env.example
└── README.md
```

## 8. Notes for going to production

- Switch to the Supabase **Session pooler** or a dedicated Postgres connection
  if you outgrow the Transaction pooler's connection limits.
- Replace `allow_origins=["*"]` in `main.py` with your actual app's origin.
- Set a strong random `JWT_SECRET_KEY` in `.env` (never commit `.env` to git).
- `FREE_DAILY_LIMIT` in `.env` controls how many free rewrites/day a non-pro user gets.
- To add a "Pro" paid tier: set `user.is_pro = 1` in the DB once payment is confirmed
  (e.g. after a successful Stripe/JazzCash webhook) — unlimited rewrites are already
  wired to skip the quota check for pro users.
- Add new tones by adding one line to `TONE_PROMPTS` in `app/ai.py` — no other
  code changes needed, `/tones` and `/rewrite` pick it up automatically.

## 9. Deploy to Vercel

1. Push this project to GitHub (`.env` is git-ignored, so your secrets stay local).
2. Go to https://vercel.com/new and import the repo. Framework preset: **Other**.
3. In **Project -> Settings -> Environment Variables** add:

   | Name | Value |
   |------|-------|
   | `GROQ_API_KEY` | your Groq key |
   | `GROQ_MODEL` | `openai/gpt-oss-120b` |
   | `JWT_SECRET_KEY` | long random string (`python -c "import secrets; print(secrets.token_urlsafe(48))"`) |
   | `DATABASE_URL` | Supabase **Transaction pooler** URI (port `6543`) |
   | `FREE_DAILY_LIMIT` | `10` |
   | `ALLOWED_ORIGINS` | `*` or your web app origin(s), comma-separated |

4. Deploy. Your API will be live at `https://<project>.vercel.app` (docs at `/docs`).

Or via CLI:
```bash
npm i -g vercel
vercel          # preview deploy
vercel --prod   # production deploy
```

Notes:
- The app refuses to start in production if `JWT_SECRET_KEY` is missing.
- On Vercel no local DB connection pool is used; use Supabase's Transaction pooler (6543).
- Tables are created automatically on the first request after a cold start.
