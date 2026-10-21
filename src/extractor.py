import json
import logging
from pathlib import Path
from PIL import Image
from .field_parser import FieldParser, FieldValue
from .ocr_engine import OCRManager, TextBlock
from .pdf_reader import PDFReader
from .utils import is_image, is_pdf, load_image

logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(
        self,
        ocr_engine: str = "easyocr",
        ocr_languages: list[str] | None = None,
        use_gpu: bool = True,
        use_vision_model: bool = False,
        vision_model_name: str = "naver-clova-ix/donut-base",
        pdf_dpi: int = 300,
        ocr_min_confidence: float = 0.3,
        line_tolerance: int = 15,
        min_label_value_gap: int = 30,
    ):
        self.use_gpu = use_gpu
        self.ocr_min_confidence = ocr_min_confidence
        self.use_vision_model = use_vision_model

        # Inicializar componentes
        self.ocr = OCRManager(engine=ocr_engine, languages=ocr_languages, use_gpu=use_gpu)
        self.pdf_reader = PDFReader(dpi=pdf_dpi)
        self.field_parser = FieldParser(line_tolerance=line_tolerance, min_label_value_gap=min_label_value_gap)

        # Modelo de visão (lazy loading)
        self._vision_model = None
        self._vision_model_name = vision_model_name

    @property
    def vision_model(self):
        if self._vision_model is None and self.use_vision_model:
            from .vision_model import VisionDocumentExtractor
            self._vision_model = VisionDocumentExtractor(model_name=self._vision_model_name, device="cuda" if self.use_gpu else "cpu")
        return self._vision_model

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
        questions: list[str] | None = None,
    ) -> "ExtractionResult":
        file_path = Path(file_path)
        if not file_path.exists(): raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        logger.info(f"Processando: {file_path}")

        if is_pdf(file_path):
            return self._extract_from_pdf(file_path, pages=pages, questions=questions)
        elif is_image(file_path):
            return self._extract_from_image(file_path, questions=questions)
        else:
            raise ValueError(f"Formato não suportado: {file_path.suffix}. Use PDF, PNG, JPG, TIFF, BMP ou WEBP.")

    def _extract_from_pdf(
        self,
        pdf_path: Path,
        pages: list[int] | None = None,
        questions: list[str] | None = None,
    ) -> "ExtractionResult":
        all_fields = []
        all_blocks = []
        page_results = []

        # Verificar se PDF tem texto embutido
        has_text = self.pdf_reader.has_embedded_text(pdf_path)

        if has_text:
            # Extrair texto nativo do PDF (mais preciso)
            logger.info("PDF com texto embutido - extraindo diretamente")
            pdf_blocks = self.pdf_reader.extract_text_blocks(pdf_path)

            for page_num, page_blocks in enumerate(pdf_blocks):
                if pages is not None and page_num not in pages: continue

                fields = self.field_parser.parse_from_pdf_blocks(page_blocks)
                all_fields.extend(fields)
                all_blocks.extend([
                        TextBlock(text=b["text"], bbox=b["bbox"], confidence=1.0)
                        for b in page_blocks
                    ])
                page_results.append({"page": page_num, "fields": fields, "source": "pdf_native"})

        # Também fazer OCR nas imagens (captura conteúdo em imagens dentro do PDF)
        pdf_images = self.pdf_reader.pdf_to_images(pdf_path)

        for page_num, page_image in enumerate(pdf_images):
            if pages is not None and page_num not in pages: continue

            # OCR
            ocr_blocks = self.ocr.extract(page_image, self.ocr_min_confidence)

            # Evitar duplicatas se já temos texto nativo
            if not has_text:
                fields = self.field_parser.parse(ocr_blocks)
                all_fields.extend(fields)
                all_blocks.extend(ocr_blocks)
                page_results.append({"page": page_num, "fields": fields, "source": "ocr"})

            # Modelo de visão
            if self.use_vision_model:
                pil_image = Image.fromarray(page_image)
                vision_fields = self._extract_with_vision(pil_image, questions=questions)
                all_fields.extend(vision_fields)

        # Deduplica campos de todas as fontes
        all_fields = self.field_parser._deduplicate(all_fields)

        return ExtractionResult(
            file_path=str(pdf_path),
            fields={fv.field: fv.value for fv in all_fields},
            field_details=all_fields,
            text_blocks=all_blocks,
            page_results=page_results,
            source_type="pdf",
        )

    def _extract_from_image(
        self,
        image_path: Path,
        questions: list[str] | None = None,
    ) -> "ExtractionResult":
        image = load_image(image_path)

        # OCR
        ocr_blocks = self.ocr.extract(image, self.ocr_min_confidence)

        # Parser espacial
        fields = self.field_parser.parse(ocr_blocks)

        # Modelo de visão
        if self.use_vision_model:
            pil_image = Image.fromarray(image)
            vision_fields = self._extract_with_vision(pil_image, questions=questions)
            fields.extend(vision_fields)

        # Deduplica
        fields = self.field_parser._deduplicate(fields)

        return ExtractionResult(
            file_path=str(image_path),
            fields={fv.field: fv.value for fv in fields},
            field_details=fields,
            text_blocks=ocr_blocks,
            page_results=[{"page": 0, "fields": fields, "source": "ocr"}],
            source_type="image",
        )

    def _extract_with_vision(
        self,
        image: Image.Image,
        questions: list[str] | None = None,
    ) -> list[FieldValue]:
        if self.vision_model is None: return []

        fields = []

        # Extração geral
        vision_fields = self.vision_model.extract_fields(image)
        for key, value in vision_fields.items():
            if not key.startswith("_"):
                fields.append(
                    FieldValue(
                        field=key,
                        value=str(value),
                        confidence=0.85,
                        source="vision_model",
                    )
                )

        # Perguntas específicas
        if questions:
            answers = self.vision_model.extract_with_questions(image, questions)
            for question, answer in answers.items():
                fields.append(FieldValue(field=question, value=answer, confidence=0.80, source="vision_model_qa"))

        return fields

    def extract_batch(self, file_paths: list[str | Path], **kwargs) -> list["ExtractionResult"]:
        results = []
        for path in file_paths:
            try:
                result = self.extract(path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Erro ao processar {path}: {e}")
                results.append(
                    ExtractionResult(
                        file_path=str(path),
                        fields={},
                        field_details=[],
                        text_blocks=[],
                        page_results=[],
                        source_type="error",
                        error=str(e),
                    )
                )
        return results

class ExtractionResult:
    def __init__(
        self,
        file_path: str,
        fields: dict[str, str],
        field_details: list[FieldValue],
        text_blocks: list[TextBlock],
        page_results: list[dict],
        source_type: str,
        error: str | None = None,
    ):
        self.file_path = file_path
        self.fields = fields
        self.field_details = field_details
        self.text_blocks = text_blocks
        self.page_results = page_results
        self.source_type = source_type
        self.error = error

    def get_field(self, field_name: str, default: str | None = None) -> str | None:
        field_lower = field_name.lower()

        # Busca exata (case-insensitive)
        for key, value in self.fields.items():
            if key.lower() == field_lower: return value

        # Busca parcial (contém)
        for key, value in self.fields.items():
            if field_lower in key.lower() or key.lower() in field_lower: return value

        return default

    def search_fields(self, query: str) -> dict[str, str]:
        query_lower = query.lower()
        return {
            key: value
            for key, value in self.fields.items()
            if query_lower in key.lower() or query_lower in value.lower()
        }

    def get_raw_text(self) -> str:
        lines = []
        sorted_blocks = sorted(self.text_blocks, key=lambda b: (b.y1, b.x1))

        current_y = -1
        current_line = []

        for block in sorted_blocks:
            if current_y >= 0 and abs(block.y1 - current_y) > 15:
                lines.append(" ".join(current_line))
                current_line = []

            current_line.append(block.text)
            current_y = block.y1

        if current_line: lines.append(" ".join(current_line))

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "source_type": self.source_type,
            "fields": self.fields,
            "field_details": [
                {
                    "field": fv.field,
                    "value": fv.value,
                    "confidence": fv.confidence,
                    "source": fv.source,
                }
                for fv in self.field_details
            ],
            "raw_text": self.get_raw_text(),
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_dataframe(self):
        import pandas as pd

        data = []
        for fv in self.field_details:
            data.append(
                {
                    "Campo": fv.field,
                    "Valor": fv.value,
                    "Confiança": f"{fv.confidence:.2f}",
                    "Fonte": fv.source,
                }
            )
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(file='{Path(self.file_path).name}', "
            f"fields={len(self.fields)}, type={self.source_type})"
        )

    def print_summary(self):
        from .utils import format_results_table

        print(f"\n{'='*60}")
        print(f"📄 Arquivo: {self.file_path}")
        print(f"📋 Tipo: {self.source_type}")
        print(f"🔍 Campos encontrados: {len(self.fields)}")
        print(f"{'='*60}")

        if self.error:
            print(f"❌ Erro: {self.error}")
            return

        print(format_results_table(self.fields))

        if self.field_details:
            print(f"\n📊 Detalhes:")
            for fv in self.field_details:
                print(
                    f"  • {fv.field}: {fv.value} "
                    f"(confiança: {fv.confidence:.0%}, fonte: {fv.source})"
                )
