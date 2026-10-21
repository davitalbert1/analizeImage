import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    text: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float = 0.0

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def __repr__(self) -> str:
        return f'TextBlock("{self.text}", bbox={self.bbox}, conf={self.confidence:.2f})'

class EasyOCREngine:
    def __init__(self, languages: list[str] | None = None, use_gpu: bool = True):
        import easyocr

        self.languages = languages or ["pt", "en"]
        self.reader = easyocr.Reader(self.languages, gpu=use_gpu)
        logger.info(f"EasyOCR inicializado: idiomas={self.languages}, gpu={use_gpu}")

    def extract(self, image: np.ndarray | str | Path, min_confidence: float = 0.3) -> list[TextBlock]:
        if isinstance(image, (str, Path)): image = str(image)

        results = self.reader.readtext(image)
        blocks = []

        for bbox_points, text, confidence in results:
            if confidence < min_confidence: continue

            # EasyOCR retorna 4 pontos [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            # Converter para (x1, y1, x2, y2)
            xs = [int(p[0]) for p in bbox_points]
            ys = [int(p[1]) for p in bbox_points]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            text = text.strip()
            if text: blocks.append(TextBlock(text=text, bbox=bbox, confidence=confidence))

        # Ordenar por posição: top→bottom, left→right
        blocks.sort(key=lambda b: (b.y1, b.x1))

        logger.info(f"OCR extraiu {len(blocks)} blocos de texto")
        return blocks

class OCRManager:
    def __init__(
        self,
        engine: str = "easyocr",
        languages: list[str] | None = None,
        use_gpu: bool = True,
    ):
        self.engine_name = engine
        if engine == "easyocr":
            self._engine = EasyOCREngine(languages=languages, use_gpu=use_gpu)
        else:
            raise ValueError(f"Motor de OCR não suportado: {engine}")

    def extract(self, image: np.ndarray | str | Path, min_confidence: float = 0.3) -> list[TextBlock]:
        return self._engine.extract(image, min_confidence=min_confidence)
