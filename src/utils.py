import logging
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    img = Image.open(path).convert("RGB")
    return np.array(img)

def is_pdf(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"

def is_image(path: str | Path) -> bool:
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    return Path(path).suffix.lower() in supported

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

def format_results_table(fields: dict[str, str]) -> str:
    if not fields: return "(nenhum campo encontrado)"

    max_key = max(len(k) for k in fields)
    lines = []
    lines.append("┌" + "─" * (max_key + 2) + "┬" + "─" * 42 + "┐")
    lines.append("│ " + "Campo".ljust(max_key) + " │ " + "Valor".ljust(40) + " │")
    lines.append("├" + "─" * (max_key + 2) + "┼" + "─" * 42 + "┤")

    for key, value in fields.items():
        val_str = str(value)[:40]
        lines.append("│ " + key.ljust(max_key) + " │ " + val_str.ljust(40) + " │")

    lines.append("└" + "─" * (max_key + 2) + "┴" + "─" * 42 + "┘")
    return "\n".join(lines)

def format_results_json(fields: dict[str, str], indent: int = 2) -> str:
    import json
    return json.dumps(fields, ensure_ascii=False, indent=indent)
