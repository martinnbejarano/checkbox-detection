"""Service settings from `CHECKBOX_*` environment variables. Imports nothing from the project."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHECKBOX_")

    locator: str = "yolox"
    classifier: str = "cnn"
    models_dir: Path = Path("models")
    onnx_threads: int = 1
    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 50_000_000
    max_pdf_pages: int = 20
    pdf_dpi: int = 300
    cv_threads: int = 1
    log_level: str = "INFO"
    static_dir: Path | None = None
