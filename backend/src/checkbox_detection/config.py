"""Service settings from `CHECKBOX_*` environment variables. Imports nothing from the project."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHECKBOX_")

    detector: str = "opencv"
    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 50_000_000
    cv_threads: int = 1
    log_level: str = "INFO"
