import os
import subprocess
import sys
import time

from sqlalchemy import create_engine, inspect, text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app.core.config import (
    CHROMA_DB_DIR,
    DATA_PATH,
    MODELS_TRAINED_DIR,
    VIEW_BUCKET_CLASSIFIER_PATH,
    VIEW_PREDICTOR_PATH,
    get_database_url,
)


def _env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _path_has_files(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def run_step(command, description: str) -> None:
    print(f"[prepare] {description}: {' '.join(command)}")
    result = subprocess.run(command, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")


def wait_for_database(timeout_seconds: int = 180) -> None:
    database_url = get_database_url()
    if not database_url:
        raise RuntimeError("DB_PASSWORD is not set; cannot prepare Docker environment")

    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            engine = create_engine(database_url, pool_pre_ping=True)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            print("[prepare] Database is ready.")
            return
        except Exception as exc:
            last_error = exc
            print(f"[prepare] Waiting for database: {exc}")
            time.sleep(5)

    raise RuntimeError(f"Database did not become ready within {timeout_seconds}s: {last_error}")


def get_video_count() -> int:
    database_url = get_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)
    inspector = inspect(engine)
    if not inspector.has_table("videos"):
        return 0

    with engine.connect() as connection:
        return int(connection.execute(text("SELECT COUNT(*) FROM videos")).scalar() or 0)


def ensure_runtime_artifacts() -> None:
    os.makedirs(MODELS_TRAINED_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    auto_load_data = _env_flag("AUTO_LOAD_DATA", "true")
    auto_train_model = _env_flag("AUTO_TRAIN_MODEL", "true")
    auto_build_chroma = _env_flag("AUTO_BUILD_CHROMA", "true")

    raw_json_path = os.path.join(BASE_DIR, "data", "raw", "output.json")
    video_count = get_video_count()
    print(f"[prepare] Current video rows: {video_count}")

    if auto_load_data and video_count == 0:
        if not os.path.exists(raw_json_path):
            raise RuntimeError(f"Missing seed data file: {raw_json_path}")
        run_step([sys.executable, "scripts/load_data_to_db.py"], "Import raw data into MySQL")
        video_count = get_video_count()
        print(f"[prepare] Video rows after import: {video_count}")

    if auto_train_model and video_count > 0:
        if not (os.path.exists(VIEW_PREDICTOR_PATH) and os.path.exists(VIEW_BUCKET_CLASSIFIER_PATH)):
            run_step([sys.executable, "scripts/train_view_predictor.py"], "Train prediction models")

    chroma_ready = _path_has_files(CHROMA_DB_DIR)
    csv_ready = os.path.exists(DATA_PATH)
    if auto_build_chroma and video_count > 0 and (not chroma_ready or not csv_ready):
        run_step([sys.executable, "scripts/topic_clustering.py"], "Build clustering data and ChromaDB")

    print("[prepare] Docker preparation finished.")


def main() -> None:
    wait_for_database(timeout_seconds=int(os.getenv("DB_WAIT_TIMEOUT", "180")))
    ensure_runtime_artifacts()


if __name__ == "__main__":
    main()
