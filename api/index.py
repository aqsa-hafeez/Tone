"""Vercel serverless entrypoint.

Vercel runs every file inside /api as a serverless function and looks for an
ASGI application called `app`. All traffic is rewritten here by vercel.json,
so FastAPI handles routing itself.
"""
import os
import sys

# Make the project root importable so `from app.main import app` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402,F401
