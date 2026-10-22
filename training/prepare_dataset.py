import argparse
import json
import sys
import os
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# UTF-8 seguro no console Windows
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

def create_example_dataset(output_dir: str):
    output = Path(output_dir)
    images_dir = output / "images"
    annotations_dir = output / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Criar anotação de exemplo
    example_annotation = {
        "image": "exemplo_exame.png",
        "fields": {
            "Paciente": "João da Silva",
            "Data": "15/10/2026",
            "Glicose": "95 mg/dL",
            "Colesterol Total": "180 mg/dL",
            "HDL": "55 mg/dL",
            "LDL": "105 mg/dL",
            "Triglicerídeos": "120 mg/dL",
            "Hemoglobina": "14.2 g/dL",
            "Hematócrito": "42.5 %",
            "Leucócitos": "7.500 /mm³",
            "Plaquetas": "250.000 /mm³",
        },
    }

    example_path = annotations_dir / "_exemplo_anotacao.json"
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(example_annotation, f, ensure_ascii=False, indent=2)

    # Criar README
    readme_content = """# Dataset de Treinamento

## Estrutura

```
training_data/
├── images/           ← Coloque as imagens dos documentos aqui
│   ├── doc001.png
│   ├── doc002.jpg
│   └── ...
└── annotations/      ← Anotações JSON correspondentes
    ├── doc001.json
    ├── doc002.json
    └── _exemplo_anotacao.json  ← Exemplo de formato
```

## Formato das Anotações

Cada arquivo JSON deve ter o mesmo nome (sem extensão) que a imagem correspondente.

Formato:
```json
{
    "image": "doc001.png",
    "fields": {
        "Nome do Campo": "Valor do campo",
        "Glicose": "95 mg/dL",
        "Data": "15/10/2026"
    }
}
```

## Como Anotar

### Opção 1: Manualmente
Crie os arquivos JSON seguindo o formato acima.

### Opção 2: Ferramenta de Anotação
```bash
py main.py label --images training_data/images --output training_data/annotations
```

## Dicas

- Use pelo menos 50-100 imagens para bons resultados
- Documente todos os campos relevantes em cada imagem
- Mantenha consistência nos nomes dos campos
- Inclua variedade de layouts e qualidades de imagem
"""

    readme_path = output / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f: f.write(readme_content)

    print(f"✅ Estrutura de dataset criada em: {output}")
    print(f"📁 {images_dir}/ (coloque as imagens aqui)")
    print(f"📁 {annotations_dir}/ (coloque as anotações JSON aqui)")
    print(f"📄 {example_path} (exemplo de formato)")
    print(f"📄 {readme_path}")
    print()
    print("Próximos passos:")
    print("1. Coloque imagens de documentos em training_data/images/")
    print("2. Anote com: py main.py label --images training_data/images")
    print("3. Treine com: py main.py train --images training_data/images --annotations training_data/annotations")

def _validar_imagem(caminho: Path) -> bool:
    try:
        with Image.open(caminho) as img: img.verify()
        with Image.open(caminho) as img:
            w, h = img.size
            return w > 0 and h > 0
    except Exception:
        return False

def _contar_pares_validos(images_dir: Path, annotations_dir: Path):
    if not images_dir.exists(): return 0, 0, []
    imagens = [f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    validos = []
    invalidos = []
    for img_path in imagens:
        ann_path = annotations_dir / f"{img_path.stem}.json"
        img_ok = _validar_imagem(img_path)
        ann_ok = ann_path.exists()
        if img_ok and ann_ok:
            validos.append(img_path.name)
        else:
            invalidos.append((img_path.name, "imagem" if not img_ok else "", "anotação" if not ann_ok else ""))
    return len(validos), len(invalidos), invalidos

def verificar_dataset(images_dir: str, annotations_dir: str):
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)

    n_validos, n_invalidos, invalidos = _contar_pares_validos(images_path, annotations_path)

    print("=" * 60)
    print("📊 Verificação do Dataset")
    print("=" * 60)
    print(f"Imagens:        {images_path}")
    print(f"Anotações:      {annotations_path}")
    print(f"Pares válidos:  {n_validos}")
    print(f"Pares inválidos: {n_invalidos}")

    if invalidos:
        print()
        print("  ⚠️  Itens com problemas:")
        for nome, img_issue, ann_issue in invalidos[:20]:
            problemas = []
            if img_issue: problemas.append("imagem inválida")
            if ann_issue: problemas.append("anotação ausente")
            print(f"     - {nome}: {', '.join(problemas)}")
        if len(invalidos) > 20:
            print(f"     ... e mais {len(invalidos) - 20} itens")

    print("=" * 60)
    return n_validos > 0

def main():
    parser = argparse.ArgumentParser(description="Preparar estrutura de dataset de treinamento")
    parser.add_argument("--output", default="./training_data", help="Diretório de saída (padrão: ./training_data)",)
    parser.add_argument(
        "--verify",
        nargs=2,
        metavar=("IMAGES_DIR", "ANNOTATIONS_DIR"),
        help="Verificar integridade de um dataset existente",
    )

    args = parser.parse_args()

    if args.verify:
        ok = verificar_dataset(args.verify[0], args.verify[1])
        sys.exit(0 if ok else 1)

    create_example_dataset(args.output)

if __name__ == "__main__":
    main()
