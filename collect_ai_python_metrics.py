#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collects Python AI repositories (ML/DL/NLP/CV/LLM), clones and computes OO metrics.
Dependencies: pip install requests radon
Optional: cloc for accurate LOC - https://github.com/AlDanial/cloc
"""

import argparse
import ast
import csv
import json
import os
import pathlib
import re
import statistics as stats
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

try:
    import requests
except Exception:
    print("Потрібно встановити 'requests' (pip install requests)", file=sys.stderr)
    raise

try:
    from radon.complexity import cc_visit
except Exception:
    cc_visit = None

EXCLUDE_DIRS = {
    "venv",
    ".venv",
    "site-packages",
    "node_modules",
    "build",
    "dist",
    "tests",
    "test",
    "docs",
    "examples",
    ".git",
    "__pycache__",
}
PY_EXT = (".py",)

SEARCH_QUERIES = [
    "language:Python topic:machine-learning -fork:true",
    "language:Python topic:deep-learning -fork:true",
    "language:Python topic:nlp -fork:true",
    "language:Python topic:llm -fork:true",
    "language:Python topic:computer-vision -fork:true",
    'language:Python "neural network" in:readme -fork:true',
]


def which(cmd: str) -> Optional[str]:
    from shutil import which as _w

    return _w(cmd)


def run(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def safe_mkdir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def iter_py_files(root: str):
    for dp, _, files in os.walk(root):
        parts = set(dp.split(os.sep))
        if parts & EXCLUDE_DIRS:
            continue
        for f in files:
            if f.endswith(PY_EXT):
                yield os.path.join(dp, f)


def estimate_loc_without_cloc(root: str) -> int:
    loc = 0
    for f in iter_py_files(root):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.strip():
                        loc += 1
        except Exception:
            pass
    return loc


def gh_search_python_ai(token: str, min_stars: int, limit: int) -> List[Dict]:
    headers = {"Authorization": f"token {token}"} if token else {}
    picked: List[Dict] = []
    seen: Set[str] = set()
    for q in SEARCH_QUERIES:
        url = "https://api.github.com/search/repositories"
        params = {
            "q": f"{q} stars:>{min_stars}",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
        }
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print("GitHub search error:", e, file=sys.stderr)
            continue
        for it in r.json().get("items", []):
            full = it["full_name"]
            if it.get("archived") or it.get("disabled"):
                continue
            if full in seen:
                continue
            seen.add(full)
            picked.append(it)
            if len(picked) >= limit:
                return picked
        time.sleep(0.5)
    return picked


def shallow_clone(clone_url: str, dest: str) -> bool:
    if os.path.exists(dest):
        return True
    cp = run(["git", "clone", "--depth", "1", clone_url, dest])
    if cp.returncode != 0:
        print("git clone failed:", clone_url, cp.stderr[:200], file=sys.stderr)
        return False
    return True


class ClassInfo:
    def __init__(self, name: str, methods: int, wmc: int, dit: int, cbo: int, rfc: int):
        self.name = name
        self.methods = methods
        self.wmc = wmc
        self.dit = dit
        self.cbo = cbo
        self.rfc = rfc


def build_cc_index(code: str):
    """Build cyclomatic complexity index using radon."""
    idx = {}
    if cc_visit is None:
        return idx
    try:
        for n in cc_visit(code):
            if hasattr(n, "lineno"):
                idx[(n.lineno, getattr(n, "col_offset", 0))] = getattr(
                    n, "complexity", 1
                )
    except Exception:
        pass
    return idx


def compute_project_metrics_py(root: str) -> Tuple[List[ClassInfo], int, int]:
    """
    Returns: (list of classes with metrics, number of relationships, number of classes).
    Relationships = unique edges of inheritance + usage.
    """
    classes: List[ClassInfo] = []
    defined_class_names: Set[str] = set()
    class_bases: Dict[str, List[str]] = {}
    usage_edges: Set[Tuple[str, str]] = set()
    inheritance_edges: Set[Tuple[str, str]] = set()

    module_classes: Dict[str, Set[str]] = {}

    # First pass: collect classes and their bases
    for f in iter_py_files(root):
        try:
            code = open(f, "r", encoding="utf-8", errors="ignore").read()
            tree = ast.parse(code)
        except Exception:
            continue
        current = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                defined_class_names.add(n.name)
                current.add(n.name)
                bases = []
                for b in n.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                class_bases[n.name] = bases
        if current:
            module_classes[f] = current

    def dit(name: str) -> int:
        """Calculate Depth of Inheritance Tree."""
        d = 0
        cur = class_bases.get(name, [])
        seen = set()
        while cur:
            nxt = []
            for b in cur:
                if b in seen:
                    continue
                seen.add(b)
                if b in class_bases:
                    nxt.extend(class_bases.get(b, []))
            if nxt:
                d += 1
            cur = nxt
        return d

    # Second pass: compute metrics
    for f in iter_py_files(root):
        try:
            code = open(f, "r", encoding="utf-8", errors="ignore").read()
            tree = ast.parse(code)
        except Exception:
            continue
        cc_idx = build_cc_index(code)
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                methods = [
                    m
                    for m in n.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]

                wmc = 0
                for m in methods:
                    key = (m.lineno, getattr(m, "col_offset", 0))
                    wmc += cc_idx.get(key, 1) if cc_visit else 1

                calls: Set[str] = set()
                refs: Set[str] = set()
                for m in methods:
                    for x in ast.walk(m):
                        if isinstance(x, ast.Call):
                            if isinstance(x.func, ast.Name):
                                calls.add(x.func.id)
                            elif isinstance(x.func, ast.Attribute):
                                calls.add(x.func.attr)
                        if isinstance(x, ast.Name):
                            refs.add(x.id)
                        elif isinstance(x, ast.Attribute):
                            refs.add(x.attr)

                rfc = len(methods) + len(calls)

                camel_like = {r for r in refs if re.match(r"[A-Z][A-Za-z0-9_]+$", r)}
                cbo = len((camel_like | (refs & defined_class_names)) - {n.name})

                for target in refs & defined_class_names:
                    if target != n.name:
                        usage_edges.add((n.name, target))

                for b in class_bases.get(n.name, []):
                    if b in defined_class_names and b != n.name:
                        inheritance_edges.add((n.name, b))

                classes.append(
                    ClassInfo(
                        name=n.name,
                        methods=len(methods),
                        wmc=wmc,
                        dit=dit(n.name),
                        cbo=cbo,
                        rfc=rfc,
                    )
                )

    edges_total = len(usage_edges | inheritance_edges)
    return classes, edges_total, len(classes)


def cloc_project_kloc(path: str) -> float:
    """Returns KLOC. Falls back to simple line counting if cloc is unavailable."""
    cloc_bin = which("cloc")
    if cloc_bin:
        cp = run(
            [
                cloc_bin,
                path,
                "--json",
                "--quiet",
                "--exclude-dir=" + ",".join(EXCLUDE_DIRS),
            ]
        )
        if cp.returncode == 0:
            try:
                data = json.loads(cp.stdout or "{}")
                total = data.get("SUM", {}).get("code", 0)
                return round(total / 1000.0, 3)
            except Exception:
                pass

    loc = estimate_loc_without_cloc(path)
    return round(loc / 1000.0, 3)


def main():
    ap = argparse.ArgumentParser(
        description="Збір Python AI репозиторіїв та метрик (CSV, укр. колонки)."
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Скільки репозиторіїв зібрати (default 50)",
    )
    ap.add_argument(
        "--min-stars", type=int, default=300, help="Мінімум зірок на GitHub"
    )
    ap.add_argument(
        "--out", type=str, default="results.csv", help="Файл для таблиці CSV"
    )
    ap.add_argument(
        "--workdir", type=str, default="work_ai_metrics", help="Робоча директорія"
    )
    ap.add_argument(
        "--no-clone",
        action="store_true",
        help="Не клонувати, аналізувати наявні каталоги в workdir/repos",
    )
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    repos_dir = os.path.join(args.workdir, "repos")
    safe_mkdir(repos_dir)

    repos: List[Dict] = []
    if not args.no_clone:
        print(f"Пошук Python AI репозиторіїв (мін. зірок: {args.min_stars}) ...")
        repos = gh_search_python_ai(token, args.min_stars, args.limit)
        print(f"Знайдено {len(repos)} репозиторіїв")
        for it in repos:
            name = it["full_name"].split("/")[-1]
            dest = os.path.join(repos_dir, name)
            if shallow_clone(it["clone_url"], dest):
                it["local_path"] = dest
            else:
                print("Пропуск (clone fail):", it["full_name"], file=sys.stderr)
    else:
        for d in sorted(os.listdir(repos_dir)):
            full = os.path.join(repos_dir, d)
            if os.path.isdir(full):
                repos.append(
                    {
                        "full_name": d,
                        "html_url": f"https://github.com/<unknown>/{d}",
                        "local_path": full,
                    }
                )

    headers = [
        "Назва проєкту",
        "Посилання",
        "Кількість рядків коду (тис.)",
        "Кількість класів",
        "Середня кількість методів у класі",
        "Глибина дерева наслідування (середня)",
        "Кількість відносин між класами (оцінка)",
        "RFC (середнє)",
        "CBO (середнє)",
        "WMC (середнє)",
    ]
    rows = [headers]

    for it in repos:
        proj = it["full_name"]
        url = it.get("html_url") or f"https://github.com/{proj}"
        path = it.get("local_path") or os.path.join(repos_dir, proj.split("/")[-1])
        print(f"→ Аналіз: {proj}")
        try:
            classes, edges_total, total_classes = compute_project_metrics_py(path)
            if total_classes == 0:
                print("  (класів не знайдено) — пропуск")
                continue
            avg_methods = stats.mean([c.methods for c in classes])
            avg_dit = stats.mean([c.dit for c in classes])
            avg_rfc = stats.mean([c.rfc for c in classes])
            avg_cbo = stats.mean([c.cbo for c in classes])
            avg_wmc = stats.mean([c.wmc for c in classes])
            kloc = cloc_project_kloc(path)

            rows.append(
                [
                    proj.split("/")[-1],
                    url,
                    f"{kloc:.3f}",
                    str(total_classes),
                    f"{avg_methods:.2f}",
                    f"{avg_dit:.2f}",
                    str(edges_total),
                    f"{avg_rfc:.2f}",
                    f"{avg_cbo:.2f}",
                    f"{avg_wmc:.2f}",
                ]
            )
        except Exception as e:
            print("  ERROR:", e, file=sys.stderr)
            continue

    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerows(rows)

    print(f"\nГотово! Таблиця збережена у: {args.out}")


if __name__ == "__main__":
    main()
