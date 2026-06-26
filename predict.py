"""
Применение обученной модели к новым парам (запись из отчёта - сотрудник).

Самодостаточный пример: загружает models/matcher_model.joblib и показывает,
как модель оценивает вероятность того, что запись и сотрудник - один человек.
Включён случай однофамильцев, чтобы показать, как модель их различает.

Запуск: python3 predict.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from features import pair_features

HERE = Path(__file__).parent
MODEL = joblib.load(HERE / "models" / "matcher_model.joblib")
CARD = json.loads((HERE / "models" / "model_card.json").read_text(encoding="utf-8"))
THRESHOLD = CARD["threshold_precision_099"]


def score(rec_fio, rec_email, rec_login, rec_manager,
          emp_fio, emp_email, emp_login, emp_manager):
    feats = [pair_features(rec_fio, rec_email, rec_login, rec_manager,
                           emp_fio, emp_email, emp_login, emp_manager)]
    return float(MODEL.predict_proba(feats)[0, 1])


def decide(p: float) -> str:
    if p >= THRESHOLD:
        return "СВЯЗАТЬ"
    if p >= 0.5:
        return "на ручную проверку"
    return "не связывать"


if __name__ == "__main__":
    print(f"Модель: {CARD['best_model']}, порог авто-связки: {THRESHOLD:.2f}\n")

    # запись из отчёта без email и без отчества (типично для ЭРА/SkillCode)
    record = ("абызов евгений", "", "", "")

    candidates = [
        ("абызов евгений игоревич", "abyzov.e@company.ru", "abyzov.e", ""),
        ("абызов евгений олегович", "abyzov.eo@company.ru", "abyzov.eo", ""),  # однофамилец
        ("кузнецова екатерина андреевна", "kuznecova.e@company.ru", "", ""),
    ]

    print(f"Запись из отчёта: {record[0]!r} (email/отчество отсутствуют)\n")
    for emp in candidates:
        p = score(*record, *emp)
        print(f"  vs {emp[0]:<32} p={p:.2f}  -> {decide(p)}")

    print("\nБез отчества и руководителя модель не может уверенно выбрать между "
          "однофамильцами - и честно отправляет на ручную проверку, "
          "а не угадывает.")
