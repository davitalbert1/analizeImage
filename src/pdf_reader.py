import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class PDFReader:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: str | Path) -> list[np.ndarray]:
        import fitz  # PyMuPDF

        pdf_path = Path(pdf_path)
        if not pdf_path.exists(): raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        images = []

        zoom = self.dpi / 72  # 72 é o DPI padrão do PDF
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)

            # Converter para numpy array RGB
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            # Se RGBA, converter para RGB
            if pix.n == 4: img = img[:, :, :3]

            images.append(img)
            logger.info(f"PDF página {page_num + 1}/{len(doc)}: {pix.width}x{pix.height}")

        doc.close()
        return images

    def extract_text_blocks(self, pdf_path: str | Path) -> list[list[dict]]:
        import fitz

        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        all_blocks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            page_blocks = []

            for block in blocks:
                if block["type"] == 0:  # Texto
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                bbox = span["bbox"]  # (x0, y0, x1, y1)
                                page_blocks.append(
                                    {
                                        "text": text,
                                        "bbox": tuple(int(c) for c in bbox),
                                        "page": page_num,
                                        "font": span.get("font", ""),
                                        "size": span.get("size", 0),
                                    }
                                )

            all_blocks.append(page_blocks)
            logger.info(f"PDF página {page_num + 1}: {len(page_blocks)} spans de texto")

        doc.close()
        return all_blocks

    def has_embedded_text(self, pdf_path: str | Path) -> bool:
        import fitz

        doc = fitz.open(str(pdf_path))
        has_text = False

        for page_num in range(min(3, len(doc))):  # Checar até 3 páginas
            page = doc[page_num]
            text = page.get_text().strip()
            if len(text) > 20:  # Mínimo de caracteres para considerar válido
                has_text = True
                break

        doc.close()
        return has_text
