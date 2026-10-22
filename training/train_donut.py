import os

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import logging
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging
from src.vision_model import DonutFineTuner

logger = logging.getLogger(__name__)

# Diretórios padrão usados quando o usuário não passa --images / --annotations
DEFAULT_IMAGES_DIR = Path("training_data/images")
DEFAULT_ANNOTATIONS_DIR = Path("training_data/annotations")

# Número mínimo de amostras padrão a gerar (garante treino + validação)
MIN_SAMPLES_PADRAO = 4

CAMPOS_PADRAO = {
    "Paciente": "Paciente Padrão",
    "Data": "15/10/2026",
    "Glicose": "90 mg/dL",
    "Colesterol Total": "180 mg/dL",
    "HDL": "55 mg/dL",
    "LDL": "105 mg/dL",
    "Triglicerídeos": "120 mg/dL",
    "Hemoglobina": "14.2 g/dL",
}

def _fonte_tamanho(tamanho):
    try:
        return ImageFont.truetype("arial.ttf", tamanho)
    except Exception:
        return ImageFont.load_default()

def gerar_imagem_padrao(caminho_saida, indice):
    largura, altura = 800, 400
    imagem = Image.new("RGB", (largura, altura), "white")
    desenho = ImageDraw.Draw(imagem)
    fonte_titulo = _fonte_tamanho(26)
    fonte_campo = _fonte_tamanho(20)

    campos = dict(CAMPOS_PADRAO)
    campos["Paciente"] = f"Paciente Padrão {indice + 1}"
    campos["Glicose"] = f"{80 + indice * 5} mg/dL"
    campos["Colesterol Total"] = f"{170 + indice * 10} mg/dL"

    desenho.text((40, 30), "EXAME PADRÃO DE TREINAMENTO", fill="black", font=fonte_titulo)

    y = 80
    for chave, valor in campos.items():
        desenho.text((40, y), f"{chave}: {valor}", fill="black", font=fonte_campo)
        y += 45

    imagem.save(caminho_saida, "PNG")

def gerar_anotacao_padrao(caminho_saida, nome_imagem, indice):
    campos = dict(CAMPOS_PADRAO)
    campos["Paciente"] = f"Paciente Padrão {indice + 1}"
    campos["Glicose"] = f"{80 + indice * 5} mg/dL"
    campos["Colesterol Total"] = f"{170 + indice * 10} mg/dL"

    anotacao = {"image": nome_imagem, "fields": campos}
    with open(caminho_saida, "w", encoding="utf-8") as f: json.dump(anotacao, f, ensure_ascii=False, indent=2)

def garantir_dados_padrao(images_dir: Path, annotations_dir: Path, usar_padrao: bool = True):
    if usar_padrao:
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

    extensoes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    imagens_existentes = (
        [p for p in images_dir.iterdir() if p.suffix.lower() in extensoes]
        if images_dir.exists()
        else []
    )

    pares_validos = 0
    for img_path in imagens_existentes:
        ann_path = annotations_dir / f"{img_path.stem}.json"
        if ann_path.exists(): pares_validos += 1

    if pares_validos > 0: return  # Já tem dados do usuário, não sobrescreve nada
    if not usar_padrao: return  # Caminho custom sem itens: deixa a validação padrão reportar o erro

    # Não há itens: gera um conjunto padrão mínimo para não quebrar o treino
    logger.info("Nenhum item de treino encontrado. Gerando imagens/anotações padrão...")
    for i in tqdm(range(MIN_SAMPLES_PADRAO), desc="Gerando itens padrão", unit="item"):
        nome_img = f"padrao_{i + 1:03d}.png"
        gerar_imagem_padrao(images_dir / nome_img, i)
        gerar_anotacao_padrao(annotations_dir / f"padrao_{i + 1:03d}.json", nome_img, i)

    print(f"✅ {MIN_SAMPLES_PADRAO} amostra(s) padrão criada(s) em {images_dir} e {annotations_dir}")

def main():
    parser = argparse.ArgumentParser(description="Treinar modelo Donut para extração de campos de documentos")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES_DIR), help=f"Diretório com imagens de treino (padrão: {DEFAULT_IMAGES_DIR})",)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS_DIR), help=f"Diretório com anotações JSON (padrão: {DEFAULT_ANNOTATIONS_DIR})")
    parser.add_argument("--model", default="naver-clova-ix/donut-base", help="Modelo base (padrão: donut-base)")
    parser.add_argument("--output-dir", default="./trained_model", help="Diretório de saída do modelo treinado")
    parser.add_argument("--epochs", type=int, default=30, help="Épocas de treino")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (padrão: 1 para Donut)")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Passos de acúmulo de gradiente (padrão: 4, batch efetivo = batch_size * steps)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--image-width", type=int, default=1280, help="Largura da imagem")
    parser.add_argument("--image-height", type=int, default=960, help="Altura da imagem")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Salvar checkpoint a cada N steps")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Proporção para validação")
    parser.add_argument("--cpu", action="store_true", help="Forçar uso de CPU")
    parser.add_argument("--seed", type=int, default=None, help="Semente para reprodutibilidade")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Desativar precisão mista (AMP)")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="Decay para EMA (0 desativa)")
    parser.add_argument("--no-grad-checkpointing", action="store_true", help="Desativar gradient checkpointing")
    parser.add_argument("--no-auto-tune", action="store_true", help="Desativar auto-tuning de recursos")
    parser.add_argument("-v", "--verbose", action="store_true", help="Modo detalhado")

    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    # Validar/garantir diretórios de treino
    images_dir = Path(args.images)
    annotations_dir = Path(args.annotations)

    usar_padrao = (str(images_dir) == str(DEFAULT_IMAGES_DIR) and str(annotations_dir) == str(DEFAULT_ANNOTATIONS_DIR))
    garantir_dados_padrao(images_dir, annotations_dir, usar_padrao=usar_padrao)

    if not images_dir.exists():
        print(f"❌ Diretório de imagens não encontrado: {images_dir}")
        sys.exit(1)

    if not annotations_dir.exists():
        print(f"❌ Diretório de anotações não encontrado: {annotations_dir}")
        sys.exit(1)

    # Inicializar fine-tuner
    device = "cpu" if args.cpu else None
    finetuner = DonutFineTuner(base_model=args.model, output_dir=args.output_dir, device=device)

    # Preparar dados
    print("📋 Preparando dados de treino...")
    training_data = finetuner.prepare_training_data(images_dir=args.images, annotations_dir=args.annotations)

    if not training_data:
        print("❌ Nenhum dado de treino válido encontrado!")
        print()
        print("Estrutura esperada:")
        print(f"{images_dir}/")
        print(f"doc001.png")
        print(f"doc002.png")
        print(f"{annotations_dir}/")
        print(f"doc001.json  →  {{\"fields\": {{\"Glicose\": \"95 mg/dL\", ...}}}}")
        print(f"doc002.json")
        sys.exit(1)

    print(f"✅ {len(training_data)} amostras preparadas")
    print()
    print("🚀 Iniciando treinamento...")
    print(f"Modelo base: {args.model}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Imagem: {args.image_width}x{args.image_height}")
    print()

    # Barra de progresso por época (visível a partir do ponto de entrada)
    pbar_epochs = tqdm(total=args.epochs, desc="Épocas", unit="época")

    # Treinar
    output = finetuner.train(
        training_data=training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        image_size=(args.image_width, args.image_height),
        save_steps=args.save_steps,
        eval_split=args.eval_split,
        seed=args.seed,
        mixed_precision=not args.no_mixed_precision,
        ema_decay=args.ema_decay,
        auto_tune=not args.no_auto_tune,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=not args.no_grad_checkpointing,
    )

    pbar_epochs.update(args.epochs)
    pbar_epochs.close()

    print()
    print("=" * 60)
    print("🎉 Treinamento concluído!")
    print(f"Modelo salvo em: {output}")
    print(f"Melhor modelo: {output}/best")
    print(f"Modelo final: {output}/final")
    print()
    print("Para usar o modelo treinado:")
    print(f"py main.py extract documento.pdf --use-ai --model {output}/best")
    print("=" * 60)

if __name__ == "__main__":
    main()
