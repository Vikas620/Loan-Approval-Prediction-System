"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration for all environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-development-secret")
    DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
