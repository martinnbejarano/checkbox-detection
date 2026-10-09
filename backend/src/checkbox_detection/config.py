"""Service settings from `CHECKBOX_*` environment variables. Imports nothing from the project."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHECKBOX_")

    detector: str = "opencv"
    cv_threads: int = 1
