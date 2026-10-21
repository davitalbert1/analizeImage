# 🔍 DocFieldExtractor

**Extração inteligente de campos e valores de qualquer tipo de documento.**

Analisa imagens de documentos (PDF, PNG, JPG, TIFF, etc.) — como resultados de exames de sangue, formulários, notas fiscais, contratos — e extrai automaticamente os campos e seus valores.

## ✨ Funcionalidades

- 📄 **Suporte a múltiplos formatos**: PDF (com texto ou escaneado), PNG, JPG, TIFF, BMP, WEBP
- 🔤 **OCR inteligente**: Extração de texto com EasyOCR (suporte a português e inglês)
- 📐 **Análise espacial**: Identifica pares campo-valor por layout (separadores, proximidade, tabelas)
- 🤖 **Modelo de IA (opcional)**: Usa Donut (Document Understanding Transformer) para extração avançada
- 🎓 **Treinamento customizado**: Fine-tune do modelo em seus documentos específicos
- 🏷️ **Ferramenta de anotação**: Interface gráfica para criar dados de treino
- 📊 **Múltiplos formatos de saída**: Tabela, JSON, CSV, DataFrame (pandas)

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────┐
│                  Documento                       │
│           (PDF, PNG, JPG, etc.)                  │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   DocumentExtractor  │   ← Orquestrador principal
          └──────────┬──────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
 ┌─────▼─────┐ ┌────▼────┐ ┌─────▼──────┐
 │ PDFReader  │ │   OCR   │ │  Modelo IA │
 │ (PyMuPDF)  │ │(EasyOCR)│ │  (Donut)   │
 └─────┬─────┘ └────┬────┘ └─────┬──────┘
       │             │             │
       └─────────────┼─────────────┘
                     │
          ┌──────────▼──────────┐
          │    FieldParser       │   ← Análise espacial
          │  (3 estratégias)     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  ExtractionResult    │   ← Resultado estruturado
          │  {campo: valor, ...} │
          └─────────────────────┘
```

### Estratégias de Extração do FieldParser

1. **Separadores inline**: Detecta padrões como `Glicose: 95 mg/dL`
2. **Proximidade espacial**: Labels à esquerda, valores à direita (com gap significativo)
3. **Estrutura tabular**: Detecta colunas alinhadas verticalmente

## 📦 Instalação

### Pré-requisitos

- Python 3.10+
- GPU NVIDIA com CUDA (opcional, mas recomendado)

### Instalar dependências

```bash
pip install -r requirements.txt
```

Dependências principais:
- `easyocr` — Motor de OCR
- `pymupdf` — Leitura de PDF
- `torch` + `torchvision` — Deep Learning
- `transformers` — Modelos de IA (Donut)
- `opencv-python` — Processamento de imagem
- `pillow` — Manipulação de imagem

## 🚀 Uso Rápido

### Via linha de comando (CLI)

```bash
# Extrair campos de um PDF
py main.py extract exame_sangue.pdf

# Extrair de uma imagem
py main.py extract resultado.png

# Buscar campo específico
py main.py extract exame.pdf --query "glicose"

# Usar modelo de IA para extração avançada
py main.py extract exame.png --use-ai

# Saída em JSON
py main.py extract exame.pdf -f json

# Salvar resultados em arquivo
py main.py extract exame.pdf -o resultado.json

# Processar pasta inteira
py main.py batch documentos/ -o resultados.json
```

### Via código Python

```python
from src.extractor import DocumentExtractor

# Criar extrator
extractor = DocumentExtractor(
    ocr_languages=["pt", "en"],
    use_gpu=True,
)

# Extrair campos
result = extractor.extract("exame_sangue.pdf")

# Ver todos os campos
print(result.fields)
# {'Glicose': '95 mg/dL', 'Colesterol Total': '180 mg/dL', ...}

# Buscar campo específico (case-insensitive, busca parcial)
glicose = result.get_field("glicose")
print(f"Glicose: {glicose}")  # "95 mg/dL"

# Buscar por palavra-chave
colesterol = result.search_fields("colesterol")
print(colesterol)  # {'Colesterol Total': '180 mg/dL', 'HDL': '55 mg/dL', ...}

# Texto raw extraído
print(result.get_raw_text())

# Exportar para JSON
print(result.to_json())

# Exportar para DataFrame (pandas)
df = result.to_dataframe()
print(df)
```

### Com modelo de IA (Donut)

```python
extractor = DocumentExtractor(
    use_vision_model=True,
    vision_model_name="naver-clova-ix/donut-base",  # ou caminho de modelo treinado
    use_gpu=True,
)

result = extractor.extract("exame.png")
result.print_summary()

# Fazer perguntas específicas ao modelo
result = extractor.extract(
    "exame.png",
    questions=[
        "Qual o nome do paciente?",
        "Qual o valor de glicose?",
        "Qual a data do exame?",
    ],
)
```

## 🎓 Treinamento de Modelo Customizado

Para melhorar a precisão em tipos específicos de documentos, você pode treinar (fine-tune) o modelo Donut nos seus próprios dados.

### Passo 1: Preparar estrutura

```bash
py training/prepare_dataset.py --output training_data
```

Isso cria a estrutura:
```
training_data/
├── images/           ← Coloque suas imagens aqui
│   ├── doc001.png
│   ├── doc002.jpg
│   └── ...
└── annotations/      ← Anotações JSON
    ├── doc001.json
    ├── doc002.json
    └── ...
```

### Passo 2: Anotar documentos

**Opção A — Ferramenta gráfica:**
```bash
py main.py label --images training_data/images --output training_data/annotations
```

A ferramenta abre uma interface onde você pode ver a imagem e adicionar campos/valores.

**Atalhos da ferramenta:**
| Atalho | Ação |
|--------|------|
| `Ctrl+N` | Adicionar campo |
| `Ctrl+S` | Salvar |
| `←` / `→` | Navegar entre imagens |
| `Delete` | Remover campo selecionado |

**Opção B — Manualmente:**

Crie arquivos JSON com o mesmo nome da imagem:
```json
// training_data/annotations/doc001.json
{
    "image": "doc001.png",
    "fields": {
        "Paciente": "João da Silva",
        "Glicose": "95 mg/dL",
        "Colesterol Total": "180 mg/dL",
        "Hemoglobina": "14.2 g/dL",
        "Data": "15/10/2026"
    }
}
```

### Passo 3: Treinar

```bash
py main.py train \
    --images training_data/images \
    --annotations training_data/annotations \
    --epochs 30 \
    --batch-size 2 \
    --lr 5e-5 \
    --output-dir ./trained_model
```

Ou diretamente:
```bash
py training/train_donut.py \
    --images training_data/images \
    --annotations training_data/annotations \
    --epochs 30
```

**Parâmetros de treino:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--epochs` | 30 | Número de épocas |
| `--batch-size` | 2 | Tamanho do batch (reduza se falta VRAM) |
| `--lr` | 5e-5 | Taxa de aprendizado |
| `--image-width` | 1280 | Largura da imagem |
| `--image-height` | 960 | Altura da imagem |
| `--warmup-steps` | 100 | Steps de warmup do optimizer |
| `--save-steps` | 500 | Salvar checkpoint a cada N steps |
| `--eval-split` | 0.1 | Proporção dos dados para validação |

### Passo 4: Usar modelo treinado

```bash
# Via CLI
py main.py extract documento.pdf --use-ai --model ./trained_model/best

# Via código
extractor = DocumentExtractor(
    use_vision_model=True,
    vision_model_name="./trained_model/best",
)
result = extractor.extract("documento.pdf")
```

### Dicas de Treinamento

- **Quantidade de dados**: Use pelo menos 50–100 imagens para bons resultados
- **Diversidade**: Inclua variações de layout, qualidade e tipos de documento
- **Consistência**: Mantenha os nomes dos campos consistentes entre anotações
- **VRAM**: Com RTX 2060 (6GB), use `batch_size=1` ou `2`
- **Épocas**: 20–50 épocas geralmente é suficiente; monitore o eval loss
- **Overfitting**: Se o train loss cai mas o eval loss sobe, reduza as épocas

## 📁 Estrutura do Projeto

```
analizeImage/
├── main.py                    # CLI principal
├── config.yaml                # Configurações
├── requirements.txt           # Dependências
├── README.md                  # Este arquivo
│
├── src/                       # Código fonte principal
│   ├── __init__.py
│   ├── extractor.py           # Orquestrador (DocumentExtractor)
│   ├── ocr_engine.py          # Motor de OCR (EasyOCR)
│   ├── pdf_reader.py          # Leitura de PDF (PyMuPDF)
│   ├── field_parser.py        # Análise espacial de campos
│   ├── vision_model.py        # Modelo de IA (Donut)
│   └── utils.py               # Utilitários
│
├── training/                  # Scripts de treinamento
│   ├── train_donut.py         # Script de fine-tuning
│   ├── prepare_dataset.py     # Preparação do dataset
│   └── label_tool.py          # Ferramenta de anotação (GUI)
│
└── examples/                  # Exemplos de uso
    └── example_usage.py
```

## 🔧 Configuração

Edite `config.yaml` para ajustar os parâmetros padrão:

```yaml
ocr_engine: "easyocr"
ocr_languages: ["pt", "en"]
use_gpu: true
pdf_dpi: 300
ocr_min_confidence: 0.3

field_parser:
  line_tolerance: 15        # px para agrupar mesma linha
  min_label_value_gap: 30   # px mínimo entre label e valor
  separators: [":", "=", " - "]
```

## 📋 Formatos de Saída

| Formato | Comando | Descrição |
|---------|---------|-----------|
| Tabela | `-f table` (padrão) | Tabela formatada no terminal |
| JSON | `-f json` | Saída em JSON |
| CSV | `-f csv` | Saída em CSV |
| Arquivo | `-o result.json` | Salva JSON em arquivo |

## 💡 Como Funciona

### Sem IA (padrão)
1. **PDF com texto embutido?** → Extrai texto diretamente (PyMuPDF)
2. **OCR** → EasyOCR extrai texto com bounding boxes
3. **Análise espacial** → FieldParser identifica pares campo:valor por:
   - Padrão separador (`:`, `=`)
   - Proximidade horizontal (label ↔ valor)
   - Estrutura tabular (colunas alinhadas)
4. **Deduplicação** → Remove duplicatas, prioriza resultados de maior confiança

### Com IA (`--use-ai`)
- Adicionalmente, usa o modelo Donut para:
  - Extração end-to-end da imagem (sem OCR)
  - Responder perguntas específicas sobre o documento
- Resultados do OCR e da IA são combinados e deduplicados

## ⚠️ Limitações

- OCR pode ter dificuldade com imagens de baixa qualidade ou manuscritas
- O modelo base Donut funciona melhor em documentos em inglês; para documentos em português, recomenda-se fine-tuning
- Documentos com layouts muito complexos (múltiplas colunas, sobreposições) podem requerer ajuste nos parâmetros do parser
- VRAM necessária: ~2GB para OCR apenas, ~4GB com modelo Donut

## 📄 Licença

Este projeto está licenciado sob a licença MIT.
