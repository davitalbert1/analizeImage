import argparse
import json
import os
import sys

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path

# Forçar encoding UTF-8 no Windows para suportar emojis/acentos
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.extractor import DocumentExtractor
from src.utils import setup_logging

def cmd_extract(args):
    extractor = DocumentExtractor(
        ocr_languages=args.languages.split(",") if args.languages else None,
        use_gpu=not args.cpu,
        use_vision_model=args.use_ai,
        vision_model_name=args.model,
        pdf_dpi=args.dpi,
        ocr_min_confidence=args.min_confidence,
    )

    result = extractor.extract(
        args.file,
        questions=args.questions if args.questions else None,
    )

    # Saída
    if args.output_format == "table":
        result.print_summary()
    elif args.output_format == "json":
        print(result.to_json())
    elif args.output_format == "csv":
        df = result.to_dataframe()
        print(df.to_csv(index=False))

    # Busca específica
    if args.query:
        print(f"\n🔎 Busca por '{args.query}':")
        matches = result.search_fields(args.query)
        if matches:
            for key, value in matches.items():
                print(f"  ✅ {key}: {value}")
        else:
            # Tentar busca por campo direto
            value = result.get_field(args.query)
            if value:
                print(f"  ✅ {args.query}: {value}")
            else:
                print(f"  ❌ Nenhum resultado encontrado para '{args.query}'")

    # Salvar em arquivo
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n💾 Resultados salvos em: {output_path}")

    return result

def cmd_batch(args):
    input_path = Path(args.directory)

    if not input_path.exists():
        print(f"❌ Diretório não encontrado: {input_path}")
        sys.exit(1)

    # Coletar arquivos
    extensions = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = [
        f
        for f in sorted(input_path.iterdir())
        if f.suffix.lower() in extensions
    ]

    if not files:
        print(f"❌ Nenhum documento encontrado em: {input_path}")
        sys.exit(1)

    print(f"📂 Encontrados {len(files)} documentos em {input_path}")

    extractor = DocumentExtractor(
        ocr_languages=args.languages.split(",") if args.languages else None,
        use_gpu=not args.cpu,
        use_vision_model=args.use_ai,
        vision_model_name=args.model,
    )

    results = extractor.extract_batch(files)

    # Resumo
    total_fields = sum(len(r.fields) for r in results)
    errors = sum(1 for r in results if r.error)

    print(f"\n{'='*60}")
    print(f"📊 Resumo do Processamento em Lote")
    print(f"{'='*60}")
    print(f"  Documentos processados: {len(results)}")
    print(f"  Total de campos extraídos: {total_fields}")
    print(f"  Erros: {errors}")

    for result in results:
        status = "❌" if result.error else "✅"
        print(f"  {status} {Path(result.file_path).name}: {len(result.fields)} campos")

    # Salvar resultados
    if args.output:
        output_path = Path(args.output)
        all_results = [r.to_dict() for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Resultados salvos em: {output_path}")

def cmd_train(args):
    from src.vision_model import DonutFineTuner

    print("🎓 Iniciando treinamento do modelo...")
    print(f"  Imagens: {args.images}")
    print(f"  Anotações: {args.annotations}")
    print(f"  Modelo base: {args.model}")
    print(f"  Saída: {args.output_dir}")

    finetuner = DonutFineTuner(
        base_model=args.model,
        output_dir=args.output_dir,
        max_length=getattr(args, "max_length", None),
    )

    # Preparar dados
    training_data = finetuner.prepare_training_data(
        images_dir=args.images,
        annotations_dir=args.annotations,
    )

    if not training_data:
        print("❌ Nenhum dado de treino encontrado!")
        print("  Certifique-se de que existem imagens e anotações JSON correspondentes.")
        sys.exit(1)

    print(f"  Dados de treino: {len(training_data)} amostras")

    # Treinar
    output = finetuner.train(
        training_data=training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=getattr(args, "grad_accum_steps", 4),
    )

    print(f"\n✅ Treinamento concluído! Modelo salvo em: {output}")
    print(f"  Para usar o modelo treinado:")
    print(f"    py main.py extract documento.pdf --use-ai --model {output}/best")

def cmd_label(args):
    from training.label_tool import run_label_tool
    run_label_tool(images_dir=args.images, output_dir=args.output)

def main():
    parser = argparse.ArgumentParser(
        description="🔍 DocFieldExtractor - Extração inteligente de campos de documentos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  py main.py extract exame.pdf                     # Extrair campos de PDF
  py main.py extract foto.png --use-ai             # Usar IA para extração
  py main.py extract exame.pdf -q "glicose"        # Buscar campo específico
  py main.py batch documentos/ -o resultados.json  # Processar pasta inteira
  py main.py train --images imgs/ --annotations ann/  # Treinar modelo custom
  py main.py label --images imgs/ --output ann/    # Anotar documentos
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Comando: extract
    p_extract = subparsers.add_parser("extract", help="Extrair campos de um documento")
    p_extract.add_argument("file", help="Caminho do arquivo (PDF ou imagem)")
    p_extract.add_argument("--use-ai", action="store_true", help="Usar modelo de IA (Donut)")
    p_extract.add_argument(
        "--model",
        default="naver-clova-ix/donut-base",
        help="Modelo de IA (padrão: donut-base)",
    )
    p_extract.add_argument("-q", "--query", help="Buscar campo específico (ex: 'glicose')")
    p_extract.add_argument(
        "--questions",
        nargs="+",
        help="Perguntas para o modelo de IA",
    )
    p_extract.add_argument(
        "-o", "--output", help="Salvar resultados em arquivo JSON"
    )
    p_extract.add_argument(
        "-f",
        "--output-format",
        choices=["table", "json", "csv"],
        default="table",
        help="Formato de saída (padrão: table)",
    )
    p_extract.add_argument(
        "--languages", default="pt,en", help="Idiomas do OCR (padrão: pt,en)"
    )
    p_extract.add_argument("--dpi", type=int, default=300, help="DPI do PDF (padrão: 300)")
    p_extract.add_argument(
        "--min-confidence", type=float, default=0.3, help="Confiança mínima OCR"
    )
    p_extract.add_argument("--cpu", action="store_true", help="Forçar uso de CPU")
    p_extract.add_argument(
        "-v", "--verbose", action="store_true", help="Modo detalhado"
    )

    # Comando: batch
    p_batch = subparsers.add_parser("batch", help="Processar múltiplos documentos")
    p_batch.add_argument("directory", help="Diretório com documentos")
    p_batch.add_argument("--use-ai", action="store_true", help="Usar modelo de IA")
    p_batch.add_argument("--model", default="naver-clova-ix/donut-base", help="Modelo de IA")
    p_batch.add_argument("-o", "--output", help="Arquivo JSON de saída")
    p_batch.add_argument("--languages", default="pt,en", help="Idiomas do OCR")
    p_batch.add_argument("--cpu", action="store_true", help="Forçar CPU")
    p_batch.add_argument("-v", "--verbose", action="store_true", help="Modo detalhado")

    # Comando: train
    p_train = subparsers.add_parser("train", help="Treinar modelo customizado")
    p_train.add_argument("--images", required=True, help="Diretório com imagens de treino")
    p_train.add_argument("--annotations", required=True, help="Diretório com anotações JSON")
    p_train.add_argument("--model", default="naver-clova-ix/donut-base", help="Modelo base")
    p_train.add_argument("--output-dir", default="./trained_model", help="Diretório de saída")
    p_train.add_argument("--epochs", type=int, default=30, help="Épocas (padrão: 30)")
    p_train.add_argument("--batch-size", type=int, default=1, help="Batch size (padrão: 1 para Donut)")
    p_train.add_argument("--grad-accum-steps", type=int, default=4, help="Passos de acúmulo de gradiente (padrão: 4)")
    p_train.add_argument("--lr", type=float, default=5e-5, help="Learning rate (padrão: 5e-5)")
    p_train.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Comprimento máximo da sequência (padrão: calculado dinamicamente)",
    )
    p_train.add_argument("-v", "--verbose", action="store_true", help="Modo detalhado")

    # Comando: label
    p_label = subparsers.add_parser("label", help="Ferramenta de anotação de documentos")
    p_label.add_argument("--images", required=True, help="Diretório com imagens para anotar")
    p_label.add_argument(
        "--output",
        default="./training_data/annotations",
        help="Diretório de saída das anotações",
    )
    p_label.add_argument("-v", "--verbose", action="store_true", help="Modo detalhado")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configurar logging
    log_level = "DEBUG" if getattr(args, "verbose", False) else "INFO"
    setup_logging(log_level)

    # Executar comando
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "label":
        cmd_label(args)

if __name__ == "__main__":
    main()
