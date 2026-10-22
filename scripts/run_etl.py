from app.core.config import get_settings
from app.repositories.database import Database
from pipelines.etl.pipeline import run_etl

settings = get_settings()
print(
	run_etl(
		settings.raw_history_data_path,
		settings.raw_current_data_path,
		Database(settings.database_path),
	)
)
