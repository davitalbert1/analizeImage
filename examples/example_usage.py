import sys
from pathlib import Path

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

def example_basic_extraction():
    from src.extractor import DocumentExtractor

    print("=" * 60)
    print("📋 Exemplo 1: Extração Básica")
    print("=" * 60)

    # Criar extrator (usa OCR + análise espacial)
    extractor = DocumentExtractor(
        ocr_languages=["pt", "en"],
        use_gpu=True,
    )

    # Extrair campos de um arquivo
    # Substitua pelo caminho do seu documento
    doc_path = "seu_documento.pdf"  # ou .png, .jpg, etc.

    if not Path(doc_path).exists():
        print(f"  ⚠️ Arquivo '{doc_path}' não encontrado.")
        print(f"  Substitua pela variável 'doc_path' com o caminho do seu documento.")
        return

    result = extractor.extract(doc_path)

    # Mostrar resultados
    result.print_summary()

    # Acessar campos específicos
    glicose = result.get_field("glicose")
    if glicose: print(f"\n🩸 Glicose: {glicose}")

    # Buscar campos por palavra-chave
    colesterol = result.search_fields("colesterol")
    if colesterol: print(f"🫀 Colesterol: {colesterol}")

def example_with_ai():
    from src.extractor import DocumentExtractor

    print("\n" + "=" * 60)
    print("🤖 Exemplo 2: Extração com Modelo de IA")
    print("=" * 60)

    # Criar extrator com IA habilitada
    extractor = DocumentExtractor(
        use_vision_model=True,
        vision_model_name="naver-clova-ix/donut-base",
        use_gpu=True,
    )

    doc_path = "seu_documento.png"

    if not Path(doc_path).exists():
        print(f"  ⚠️ Arquivo '{doc_path}' não encontrado.")
        return

    # Extração geral
    result = extractor.extract(doc_path)
    result.print_summary()

    # Fazer perguntas específicas
    result_qa = extractor.extract(
        doc_path,
        questions=["Qual o nome do paciente?", "Qual o valor de glicose?", "Qual a data do exame?"],
    )

    print("\n📝 Respostas do modelo de IA:")
    for field, value in result_qa.fields.items(): print(f"  • {field}: {value}")

def example_batch_processing():
    from src.extractor import DocumentExtractor

    print("\n" + "=" * 60)
    print("📂 Exemplo 3: Processamento em Lote")
    print("=" * 60)

    extractor = DocumentExtractor()

    docs_dir = Path("documentos")
    if not docs_dir.exists():
        print(f"  ⚠️ Diretório '{docs_dir}' não encontrado.")
        return

    # Coletar arquivos
    files = list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.png"))

    if not files:
        print(f"  ⚠️ Nenhum documento encontrado em '{docs_dir}'")
        return

    # Processar todos
    results = extractor.extract_batch(files)

    # Resumo
    for result in results:
        print(f"\n📄 {Path(result.file_path).name}:")
        for field, value in result.fields.items(): print(f"    {field}: {value}")

def example_programmatic_api():
    from src.extractor import DocumentExtractor

    print("\n" + "=" * 60)
    print("🔧 Exemplo 4: API Programática")
    print("=" * 60)

    extractor = DocumentExtractor()

    doc_path = "seu_documento.pdf"
    if not Path(doc_path).exists():
        print(f"  ⚠️ Arquivo '{doc_path}' não encontrado.")
        print()
        print("  Código de exemplo:")
        print()
        print('    extractor = DocumentExtractor()')
        print('    result = extractor.extract("exame.pdf")')
        print()
        print('    # Acessar como dicionário')
        print('    campos = result.fields')
        print('    print(campos)  # {"Glicose": "95 mg/dL", ...}')
        print()
        print('    # Buscar campo específico')
        print('    valor = result.get_field("glicose")')
        print('    print(f"Glicose: {valor}")')
        print()
        print('    # Exportar para JSON')
        print('    json_str = result.to_json()')
        print()
        print('    # Exportar para DataFrame (pandas)')
        print('    df = result.to_dataframe()')
        print('    print(df)')
        print()
        print('    # Texto raw')
        print('    texto = result.get_raw_text()')
        return

    result = extractor.extract(doc_path)

    # Acessar como dicionário
    campos = result.fields
    print(f"  Campos: {campos}")

    # Exportar para JSON
    print(f"\n  JSON:\n{result.to_json()}")

    # Exportar para DataFrame
    df = result.to_dataframe()
    print(f"\n  DataFrame:\n{df}")

def example_custom_trained_model():
    from src.extractor import DocumentExtractor

    print("\n" + "=" * 60)
    print("🎓 Exemplo 5: Modelo Treinado Customizado")
    print("=" * 60)

    model_path = "./trained_model/best"

    if not Path(model_path).exists():
        print(f"  ⚠️ Modelo treinado não encontrado em '{model_path}'")
        print()
        print("  Para treinar um modelo:")
        print("    1. Prepare dados: py training/prepare_dataset.py")
        print("    2. Anote imagens: py main.py label --images training_data/images")
        print("    3. Treine: py main.py train --images training_data/images --annotations training_data/annotations")
        print("    4. Use: py main.py extract doc.pdf --use-ai --model trained_model/best")
        return

    extractor = DocumentExtractor(use_vision_model=True, vision_model_name=model_path, use_gpu=True)

    result = extractor.extract("seu_documento.pdf")
    result.print_summary()

if __name__ == "__main__":
    print("🔍 DocFieldExtractor - Exemplos de Uso")
    print()

    example_basic_extraction()
    example_programmatic_api()
    # Descomente para testar com IA:
    # example_with_ai()
    # example_batch_processing()
    # example_custom_trained_model()
