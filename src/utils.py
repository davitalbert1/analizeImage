import logging
import sys
import os

# Configuração de alocação de memória CUDA para evitar fragmentação de VRAM (OOM)
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import psutil
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# UTF-8 seguro no console Windows (evita UnicodeEncodeError com emojis/acentos)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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
    # Suprimir logs HTTP verbosos do httpx (INFO-level HEAD/GET requests
    # para HuggingFace, incluindo respostas 404/307 de arquivos que não
    # existem no modelo, como audio_tokenizer_config.json,
    # processor_config.json, model.safetensors, generation_config.json).
    logging.getLogger("httpx").setLevel(logging.WARNING)

def profile_system_resources() -> dict:
    info = {
        "cpu": {
            "physical_cores": os.cpu_count() or 1,
            "logical_cores": os.cpu_count() or 1,
            "usage_percent": 0.0,
        },
        "ram": {
            "total_gb": 0.0,
            "available_gb": 0.0,
            "used_percent": 0.0,
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "name": "CPU Only",
            "capability": None,
            "total_vram_gb": 0.0,
            "free_vram_gb": 0.0,
            "bf16_native": False,
            "fp16_native": False,
        },
    }

    if psutil:
        try:
            info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False) or os.cpu_count() or 1
            info["cpu"]["logical_cores"] = psutil.cpu_count(logical=True) or os.cpu_count() or 1
            info["cpu"]["usage_percent"] = psutil.cpu_percent(interval=None)

            mem = psutil.virtual_memory()
            info["ram"]["total_gb"] = round(mem.total / (1024 ** 3), 2)
            info["ram"]["available_gb"] = round(mem.available / (1024 ** 3), 2)
            info["ram"]["used_percent"] = mem.percent
        except Exception as e:
            logger.debug(f"Falha ao ler métricas de sistema com psutil: {e}")

    if torch.cuda.is_available():
        try:
            dev_idx = 0
            props = torch.cuda.get_device_properties(dev_idx)
            info["gpu"]["name"] = props.name
            info["gpu"]["capability"] = (props.major, props.minor)
            info["gpu"]["total_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)

            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
            info["gpu"]["free_vram_gb"] = round(free_bytes / (1024 ** 3), 2)

            info["gpu"]["fp16_native"] = props.major >= 7
            info["gpu"]["bf16_native"] = props.major >= 8
        except Exception as e:
            logger.debug(f"Falha ao obter dados de CUDA: {e}")

    return info

def get_optimized_training_config(current_batch_size: int = 2, current_workers: int = 0) -> dict:
    res = profile_system_resources()
    gpu = res["gpu"]
    cpu = res["cpu"]
    ram = res["ram"]

    if gpu["available"]:
        if gpu["bf16_native"]:
            recommended_dtype = torch.bfloat16
            precision_str = "bfloat16"
            use_scaler = False
        else:
            recommended_dtype = torch.float16
            precision_str = "float16"
            use_scaler = True
    else:
        recommended_dtype = torch.float32
        precision_str = "float32"
        use_scaler = False

    if gpu["available"]:
        total_vram = gpu["total_vram_gb"]
        # Donut Swin-B requer alta quantidade de memória para resoluções de documento (ex: 1280x960).
        # Em GPUs de 6GB (como RTX 2060), batch_size=1 é obrigatório para evitar OOM de CUDA.
        if total_vram >= 24:
            recommended_batch_size = 8
        elif total_vram >= 16:
            recommended_batch_size = 4
        elif total_vram >= 8:
            recommended_batch_size = 2
        elif total_vram >= 6:
            recommended_batch_size = 1
        elif total_vram >= 4:
            recommended_batch_size = 1
        else:
            recommended_batch_size = 1
    else:
        if ram["available_gb"] >= 16:
            recommended_batch_size = 2
        elif ram["available_gb"] >= 8:
            recommended_batch_size = 1
        else:
            recommended_batch_size = 1

    # Quando auto_tune está ativo, não deve aumentar arbitrariamente além do limite seguro de VRAM
    # mas respeita batch menor se o usuário pediu menor
    if current_batch_size > 0:
        recommended_batch_size = min(recommended_batch_size, current_batch_size) if gpu["available"] and total_vram <= 8 else recommended_batch_size

    logical_cores = cpu["logical_cores"]
    avail_ram = ram["available_gb"]
    # No Windows e PyTorch, múltiplos workers no DataLoader multiplicam uso de RAM por fork/spawn.
    # Recomendamos 0 workers no Windows ou limite baixo para não esgotar a RAM do sistema.
    if sys.platform == "win32" or avail_ram < 4.0:
        recommended_workers = 0
    elif avail_ram < 8.0:
        recommended_workers = min(2, max(1, logical_cores // 4))
    else:
        recommended_workers = min(2, max(1, logical_cores // 2))

    if avail_ram >= 16.0:
        recommended_cache_size = 1500
    elif avail_ram >= 8.0:
        recommended_cache_size = 800
    elif avail_ram >= 4.0:
        recommended_cache_size = 300
    else:
        recommended_cache_size = 0

    config = {
        "resources": res,
        "mixed_precision_dtype": recommended_dtype,
        "precision_str": precision_str,
        "use_grad_scaler": use_scaler,
        "recommended_batch_size": recommended_batch_size,
        "recommended_workers": recommended_workers,
        "recommended_cache_size": recommended_cache_size,
    }
    return config

def print_resource_report(config: dict | None = None):
    if config is None:
        config = get_optimized_training_config()

    res = config["resources"]
    gpu = res["gpu"]
    cpu = res["cpu"]
    ram = res["ram"]

    lines = [
        "=" * 70,
        " [DIAGNÓSTICO DE HARDWARE E RECURSOS DO SISTEMA]",
        "=" * 70,
        f" * CPU: {cpu['physical_cores']} núcleos físicos, {cpu['logical_cores']} threads (Uso atual: {cpu['usage_percent']}%)",
        f" * RAM: {ram['total_gb']} GB total ({ram['available_gb']} GB disponíveis - {100 - ram['used_percent']:.1f}% livre)",
    ]

    if gpu["available"]:
        cap = gpu["capability"]
        cap_str = f"SM {cap[0]}.{cap[1]}" if cap else "N/A"
        lines.extend([
            f" * GPU: {gpu['name']} ({cap_str})",
            f" * VRAM: {gpu['total_vram_gb']} GB total ({gpu['free_vram_gb']} GB livres)",
            f" * Tensor Cores: FP16 Nativo={gpu['fp16_native']} | BF16 Nativo={gpu['bf16_native']}",
        ])
    else:
        lines.append(" * GPU: Não detectada (Modo CPU)")

    lines.extend([
        "-" * 70,
        " [CONFIGURAÇÃO RECOMENDADA PELO AUTO-TUNER]",
        f" * Batch Size Sugerido: {config['recommended_batch_size']}",
        f" * Workers DataLoader: {config['recommended_workers']}",
        f" * Precisão AMP: {config['precision_str'].upper()}",
        f" * Cache em RAM: {config['recommended_cache_size']} amostras",
        "=" * 70,
    ])

    report_str = "\n".join(lines)
    logger.info("\n" + report_str)
    try:
        print(report_str)
    except Exception:
        pass
    return report_str

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
