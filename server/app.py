"""Multi-mode deployment entrypoint."""

from data_triage_env.server import app


def main() -> None:
    """CLI script entrypoint used by multi-mode validators."""
    # Intentionally lightweight: app serving is delegated to ASGI runners.
    return None
