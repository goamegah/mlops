"""Journal de predictions en base (SQLAlchemy 2.0 + MySQL).

Trace chaque appel /predict (tracabilite / audit) et permet d'enregistrer des
feedbacks (verite terrain). Les tables sont creees au demarrage de l'API via
``init_db()``. La connexion est lue depuis ``DATABASE_URL`` (config / .env).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import JSON, ForeignKey, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from bank_marketing.config import DATABASE_URL

# create_engine ne se connecte pas tout de suite : importer ce module n'exige
# pas que MySQL soit deja demarre (la connexion a lieu au 1er acces / init_db).
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Prediction(Base):
    """Une prediction servie par l'API."""

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[dt.datetime] = mapped_column(default=_now)
    features: Mapped[dict[str, Any]] = mapped_column(JSON)
    prediction: Mapped[int]
    probability: Mapped[float]
    model_version: Mapped[str] = mapped_column(String(50), default="unknown")


class Feedback(Base):
    """Verite terrain associee a une prediction (le client a-t-il souscrit ?)."""

    __tablename__ = "feedbacks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(String(32), ForeignKey("predictions.id"))
    actual: Mapped[int]
    created_at: Mapped[dt.datetime] = mapped_column(default=_now)


def init_db() -> None:
    """Creer les tables si elles n'existent pas (idempotent)."""
    Base.metadata.create_all(engine)


def save_prediction(
    features: dict[str, Any], prediction: int, probability: float, model_version: str
) -> str:
    """Enregistrer une prediction et renvoyer son identifiant."""
    pred_id = _uuid()
    with Session(engine) as session:
        session.add(
            Prediction(
                id=pred_id,
                features=features,
                prediction=prediction,
                probability=probability,
                model_version=model_version,
            )
        )
        session.commit()
    return pred_id


def save_feedback(prediction_id: str, actual: int) -> str:
    """Enregistrer un feedback rattache a une prediction."""
    fb_id = _uuid()
    with Session(engine) as session:
        session.add(Feedback(id=fb_id, prediction_id=prediction_id, actual=actual))
        session.commit()
    return fb_id


def list_predictions(limit: int = 50) -> list[dict[str, Any]]:
    """Renvoyer les ``limit`` dernieres predictions (avec feedback eventuel)."""
    with Session(engine) as session:
        preds = session.scalars(
            select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
        ).all()
        feedbacks = {f.prediction_id: f.actual for f in session.scalars(select(Feedback)).all()}
        return [
            {
                "id": p.id,
                "created_at": p.created_at.isoformat(),
                "prediction": p.prediction,
                "probability": p.probability,
                "model_version": p.model_version,
                "actual": feedbacks.get(p.id),
                **p.features,
            }
            for p in preds
        ]
