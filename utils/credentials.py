# utils/credentials.py
"""
Credential manager — stores username/password pairs keyed by site/app name.

Storage: ~/.config/gui-agent/credentials.json  (plaintext, local only)
Security note: do not use for high-value accounts on shared machines.

Planner uses the token format:  {{cred:github.com:username}}
                                 {{cred:github.com:password}}

action_agent substitutes these tokens before typing.
"""
import json
import re
from pathlib import Path
from typing import Optional

_CRED_DIR  = Path.home() / ".config" / "gui-agent"
_CRED_FILE = _CRED_DIR / "credentials.json"


def _load() -> dict:
    if not _CRED_FILE.exists():
        return {}
    try:
        return json.loads(_CRED_FILE.read_text())
    except Exception:
        return {}


def _save(data: dict):
    _CRED_DIR.mkdir(parents=True, exist_ok=True)
    _CRED_FILE.write_text(json.dumps(data, indent=2))


def _best_key(store: dict, site: str) -> Optional[str]:
    """Fuzzy-match site name against stored keys (case-insensitive substring)."""
    site_lower = site.lower()
    # Exact match first
    if site_lower in store:
        return site_lower
    # Substring match
    for key in store:
        if site_lower in key or key in site_lower:
            return key
    return None


def get(site: str, field: str) -> Optional[str]:
    """Return stored username or password for a site. Returns None if not found."""
    store = _load()
    key = _best_key(store, site)
    if key is None:
        return None
    return store[key].get(field.lower())


def set_credential(site: str, username: str, password: str):
    """Store or update credentials for a site."""
    store = _load()
    store[site.lower()] = {"username": username, "password": password}
    _save(store)


def delete(site: str):
    """Remove credentials for a site."""
    store = _load()
    key = _best_key(store, site)
    if key and key in store:
        del store[key]
        _save(store)


def list_sites() -> list:
    """Return all stored site names."""
    return sorted(_load().keys())


# Token pattern: {{cred:github.com:username}}
_TOKEN_RE = re.compile(r'\{\{cred:([^:}]+):([^}]+)\}\}')


def substitute(text: str) -> str:
    """Replace {{cred:site:field}} tokens in text with stored values.
    Tokens with no stored credential are left as-is (safe no-op).
    """
    def _replace(m: re.Match) -> str:
        site, field = m.group(1), m.group(2)
        value = get(site, field)
        if value is None:
            return m.group(0)   # leave token unchanged — don't leak a blank password
        return value

    return _TOKEN_RE.sub(_replace, text)


def has_tokens(text: str) -> bool:
    """Return True if text contains any credential tokens."""
    return bool(_TOKEN_RE.search(text))
