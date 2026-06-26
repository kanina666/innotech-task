from __future__ import annotations

from rapidfuzz import fuzz, distance

# порядок важен - в этом же порядке идут колонки в X и подписи на графике importance
FEATURE_NAMES = [
    "fio_token_set",
    "fio_token_sort",
    "fio_jaro_winkler",
    "fio_levenshtein",
    "fio_overlap",
    "fio_containment",   # одно ФИО вложено в другое (когда нет отчества)
    "email_exact",
    "login_match",
    "manager_match",
    "len_ratio",
]

_TRANSLIT = {"a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е", "z": "з",
             "i": "и", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
             "r": "р", "s": "с", "t": "т", "u": "у", "f": "ф", "h": "х", "y": "ы"}


def _translit(s):
    return "".join(_TRANSLIT.get(c, c) for c in s.lower())


def _email_local(email):
    return email.split("@", 1)[0] if "@" in email else ""


def pair_features(rec_fio, rec_email, rec_login, rec_manager,
                  emp_fio, emp_email, emp_login, emp_manager):
    fio_ts = fuzz.token_set_ratio(rec_fio, emp_fio) / 100.0
    fio_sort = fuzz.token_sort_ratio(rec_fio, emp_fio) / 100.0
    fio_jw = distance.JaroWinkler.similarity(rec_fio, emp_fio)
    fio_lev = 1.0 - distance.Levenshtein.normalized_distance(rec_fio, emp_fio)

    rt, et = set(rec_fio.split()), set(emp_fio.split())
    inter = len(rt & et)
    overlap = inter / max(1, len(rt | et))
    containment = inter / max(1, min(len(rt), len(et)))

    email_exact = 1.0 if (rec_email and rec_email == emp_email) else 0.0

    login = rec_login or ""
    emp_local = _email_local(emp_email)
    if login and emp_local:
        # логин либо похож на левую часть email, либо на транслит ФИО
        login_match = max(
            fuzz.ratio(login.lower(), emp_local) / 100.0,
            fuzz.ratio(_translit(login), emp_fio.replace(" ", "")) / 100.0,
        )
    else:
        login_match = 0.0

    mgr = (fuzz.token_set_ratio(rec_manager, emp_manager) / 100.0
           if rec_manager and emp_manager else 0.0)

    lr = min(len(rec_fio), len(emp_fio)) / max(1, max(len(rec_fio), len(emp_fio)))

    return [fio_ts, fio_sort, fio_jw, fio_lev, overlap, containment,
            email_exact, login_match, mgr, lr]
