import argparse
import json
from pathlib import Path

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
    print(f"   📁 {images_dir}/ (coloque as imagens aqui)")
    print(f"   📁 {annotations_dir}/ (coloque as anotações JSON aqui)")
    print(f"   📄 {example_path} (exemplo de formato)")
    print(f"   📄 {readme_path}")
    print()
    print("Próximos passos:")
    print("  1. Coloque imagens de documentos em training_data/images/")
    print("  2. Anote com: py main.py label --images training_data/images")
    print("  3. Treine com: py main.py train --images training_data/images --annotations training_data/annotations")

def main():
    parser = argparse.ArgumentParser(description="Preparar estrutura de dataset de treinamento")
    parser.add_argument("--output", default="./training_data", help="Diretório de saída (padrão: ./training_data)",)

    args = parser.parse_args()
    create_example_dataset(args.output)

if __name__ == "__main__":
    main()
