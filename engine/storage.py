"""
TRAJNO SHRANJEVANJE na GitHub (veja "data", da se aplikacija ob shranjevanju ne ponovno zažene)
Potrebno v Streamlit Secrets / okoljskih spremenljivkah:
    GH_TOKEN = "github_pat_..."         (fine-grained token z dovoljenjem Contents: Read and write)
    GH_REPO  = "uporabnik/repozitorij"
Brez tokena vse deluje naprej, le shranjevanje je samo začasno.
"""
import base64
import json
import os

import requests

API = "https://api.github.com"
BRANCH = "data"


def _cfg():
    token = os.environ.get("GH_TOKEN")
    repo = os.environ.get("GH_REPO")
    try:
        import streamlit as st
        token = token or st.secrets.get("GH_TOKEN")
        repo = repo or st.secrets.get("GH_REPO")
    except Exception:
        pass
    return token, repo


def enabled():
    t, r = _cfg()
    return bool(t and r)


def _h(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _ensure_branch(token, repo):
    r = requests.get(f"{API}/repos/{repo}/git/ref/heads/{BRANCH}", headers=_h(token), timeout=15)
    if r.status_code == 200:
        return
    info = requests.get(f"{API}/repos/{repo}", headers=_h(token), timeout=15).json()
    base = info.get("default_branch", "main")
    sha = requests.get(f"{API}/repos/{repo}/git/ref/heads/{base}", headers=_h(token), timeout=15).json()["object"]["sha"]
    requests.post(f"{API}/repos/{repo}/git/refs", headers=_h(token), timeout=15,
                  json={"ref": f"refs/heads/{BRANCH}", "sha": sha}).raise_for_status()


def read_file(path):
    """Vrne (besedilo, sha) ali (None, None)."""
    token, repo = _cfg()
    if not (token and repo):
        return None, None
    r = requests.get(f"{API}/repos/{repo}/contents/{path}", params={"ref": BRANCH},
                     headers=_h(token), timeout=15)
    if r.status_code != 200:
        return None, None
    j = r.json()
    return base64.b64decode(j["content"]).decode("utf-8"), j["sha"]


def write_file(path, text, message):
    token, repo = _cfg()
    if not (token and repo):
        raise RuntimeError("GH_TOKEN / GH_REPO nista nastavljena")
    _ensure_branch(token, repo)
    _, sha = read_file(path)
    body = {"message": message, "branch": BRANCH,
            "content": base64.b64encode(text.encode("utf-8")).decode()}
    if sha:
        body["sha"] = sha
    requests.put(f"{API}/repos/{repo}/contents/{path}", headers=_h(token), json=body,
                 timeout=20).raise_for_status()


def load_settings(default_path):
    text, _ = read_file("settings.json")
    src = "github"
    if text is None:
        with open(default_path, encoding="utf-8") as f:
            text = f.read()
        src = "privzeto"
    s = json.loads(text)
    s.setdefault("overrides", {})
    s.setdefault("risk", {"target_vol": 0.15, "max_lev": 3.0, "cost": 0.001})
    s.setdefault("paper", {"enabled": False, "only_edge": True, "capital_use": 1.0})
    s.setdefault("edge", [])
    s.setdefault("custom_strategies", {})
    s.setdefault("lab_saved", {})
    return s, src


def save_settings(settings):
    write_file("settings.json", json.dumps(settings, ensure_ascii=False, indent=1, default=str),
               "posodobitev nastavitev iz aplikacije")
