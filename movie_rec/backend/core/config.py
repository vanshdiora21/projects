import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        TMDB_API_KEY = os.getenv("TMDB_API_KEY")
        if not TMDB_API_KEY:
            raise RuntimeError("TMDB_API_KEY is not set.")

        self.DEBUG: bool = True

        print(f"[DEBUG] TMDB_API_KEY = {self.TMDB_API_KEY}")  # ✅ Will now print on init

settings = Settings()
