"""Object to return important paths."""
from pathlib import Path
from types import SimpleNamespace

paths = SimpleNamespace()
paths.root = Path(__file__).parents[1]
paths.data = paths.root / "data"
paths.models = paths.root / "models"
paths.figures = paths.root / "figures"
