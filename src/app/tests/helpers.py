"""Helpers shared by the Streamlit app tests."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
PAGE_TIMEOUT = 120


class FakeSessionState(dict):
    """Dict with attribute access, standing in for st.session_state outside of a Streamlit run."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def page(name) -> AppTest:
    path = APP_DIR / "pages" / name if name != "welcome" else APP_DIR / "Welcome to the ACFX.py"
    return AppTest.from_file(str(path), default_timeout=PAGE_TIMEOUT)


def with_state(app: AppTest, state: dict) -> AppTest:
    for key, value in state.items():
        app.session_state[key] = value
    return app
