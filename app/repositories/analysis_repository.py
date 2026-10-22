from app.models.domain import Feedback, PredictionResult
from app.repositories.database import Database


class AnalysisRepository:
    def __init__(self, database: Database):
        self.database = database

    def save_prediction(self, analysis_id: str, prediction: PredictionResult) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO predictions(
                    analysis_id, customer_id, probability, risk_band, model_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    prediction.customer_id,
                    prediction.churn_probability,
                    prediction.risk_band,
                    prediction.model_version,
                ),
            )

    def create_ticket(self, analysis_id: str, customer_id: str, title: str, body: str) -> int:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO retention_tickets(analysis_id, customer_id, title, body)
                VALUES (?, ?, ?, ?)
                """,
                (analysis_id, customer_id, title, body),
            )
            return int(cursor.lastrowid)

    def save_feedback(self, feedback: Feedback) -> int:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback(analysis_id, rating, comment)
                VALUES (?, ?, ?)
                """,
                (feedback.analysis_id, feedback.rating, feedback.comment),
            )
            return int(cursor.lastrowid)

    def get_customer_id_by_analysis(self, analysis_id: str) -> str | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT customer_id
                FROM predictions
                WHERE analysis_id = ?
                LIMIT 1
                """,
                (analysis_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["customer_id"])
