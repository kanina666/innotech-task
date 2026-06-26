from __future__ import annotations

import random

from rapidfuzz import fuzz

from features import pair_features, FEATURE_NAMES

# справочники для генерации синтетики (реальной разметки нет)
FIRST = ["иван", "петр", "сергей", "алексей", "дмитрий", "андрей", "мария",
         "елена", "ольга", "анна", "наталья", "татьяна", "михаил", "николай",
         "екатерина", "константин", "лариса", "евгений"]
LAST = ["иванов", "петров", "смирнов", "кузнецов", "соколов", "попов", "лебедев",
        "козлов", "новиков", "морозов", "волков", "федоров", "абызов", "абакумов",
        "громов", "зацепин"]
PATR = ["иванович", "петрович", "сергеевич", "алексеевна", "дмитриевна",
        "андреевна", "михайлович", "николаевна", "игоревич", "евгеньевна"]

_FEM_NAMES = {"мария", "елена", "ольга", "анна", "наталья", "татьяна",
              "екатерина", "лариса"}

_TR = {"а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","з":"z","и":"i","к":"k",
       "л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u",
       "ф":"f","х":"h","ц":"c","ы":"y","й":"y"}


def _tr(s):
    return "".join(_TR.get(c, "") for c in s.lower())


def _gender_fix(last, first, patr):
    # чтобы женские ФИО не выглядели как "иванов мария" с мужским отчеством
    if first in _FEM_NAMES:
        if last.endswith(("ов", "ев", "ин")):
            last += "а"
        if not patr.endswith("на"):
            patr = patr.replace("ич", "на")
    return last, first, patr


def make_employees(n, seed=7):
    rnd = random.Random(seed)
    emps = []
    for i in range(n):
        last, first, patr = _gender_fix(rnd.choice(LAST), rnd.choice(FIRST),
                                        rnd.choice(PATR))
        fio = " ".join(sorted([first, last, patr]))  # токены сортируем, порядок не важен
        suffix = rnd.randint(0, 9) if rnd.random() < 0.4 else ""
        login = f"{_tr(last)}.{_tr(first)[:1]}{suffix}"
        manager = " ".join(sorted(_gender_fix(rnd.choice(LAST), rnd.choice(FIRST),
                                              rnd.choice(PATR))))
        emps.append({"id": f"E{i:04d}", "fio": fio, "email": f"{login}@company.ru",
                     "login": login, "manager": manager})
    return emps


def _typo(s, rnd):
    if len(s) < 4:
        return s
    i = rnd.randint(1, len(s) - 2)
    op = rnd.choice(["swap", "drop", "dup"])
    if op == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "drop":
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def _noisy_record(emp, rnd):
    fio = emp["fio"]
    if rnd.random() < 0.30:            # часто в отчётах нет отчества
        toks = fio.split()
        if len(toks) == 3:
            fio = " ".join(toks[:2])
    if rnd.random() < 0.20:
        fio = fio.replace("е", "ё", 1)
    if rnd.random() < 0.18:
        fio = _typo(fio, rnd)
    if rnd.random() < 0.25:
        toks = fio.split()
        rnd.shuffle(toks)
        fio = " ".join(toks)

    has_email = rnd.random() < 0.45
    email = emp["email"] if has_email else ""
    login = emp["login"] if (not has_email and rnd.random() < 0.5) else ""
    manager = emp["manager"] if rnd.random() < 0.5 else ""
    return {"fio": fio, "email": email, "login": login, "manager": manager}


def _feats(rec, emp):
    return pair_features(rec["fio"], rec["email"], rec["login"], rec["manager"],
                         emp["fio"], emp["email"], emp["login"], emp["manager"])


def build_dataset(n_emp=400, neg_per_pos=3, seed=11):
    rnd = random.Random(seed)
    emps = make_employees(n_emp, seed)

    # индекс токен -> сотрудники, чтобы быстро находить однофамильцев
    by_token = {}
    for e in emps:
        for tok in e["fio"].split():
            by_token.setdefault(tok, []).append(e)

    X, y = [], []
    for emp in emps:
        for _ in range(rnd.randint(1, 3)):
            rec = _noisy_record(emp, rnd)
            X.append(_feats(rec, emp))
            y.append(1)

            # негативы: сначала однофамильцы/похожие, потом добор случайными
            cand_ids = set()
            for tok in rec["fio"].split():
                for o in by_token.get(tok, []):
                    if o["id"] != emp["id"]:
                        cand_ids.add(o["id"])
            cands = [e for e in emps if e["id"] in cand_ids]
            cands.sort(key=lambda o: fuzz.token_set_ratio(rec["fio"], o["fio"]),
                       reverse=True)
            negs = cands[:neg_per_pos]
            while len(negs) < neg_per_pos:
                o = rnd.choice(emps)
                if o["id"] != emp["id"] and o not in negs:
                    negs.append(o)
            for o in negs:
                X.append(_feats(rec, o))
                y.append(0)

    return X, y, FEATURE_NAMES, {"n_emp": n_emp, "neg_per_pos": neg_per_pos}
