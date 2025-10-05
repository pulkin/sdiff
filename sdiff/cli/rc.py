from pathlib import Path
import importlib.util


if (p := Path.home() / ".sdiffrc").exists():
    spec = importlib.util.spec_from_file_location("_sdiffrc", p)
    importlib.util.module_from_spec(spec)
