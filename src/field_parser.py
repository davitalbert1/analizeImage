import logging
import re
from dataclasses import dataclass
from .ocr_engine import TextBlock

logger = logging.getLogger(__name__)

@dataclass
class FieldValue:
    field: str
    value: str
    confidence: float = 0.0
    source: str = ""  # Estratégia que encontrou este par

    def __repr__(self) -> str:
        return f'FieldValue("{self.field}" = "{self.value}", conf={self.confidence:.2f})'

class FieldParser:
    def __init__(
        self,
        line_tolerance: int = 15,
        min_label_value_gap: int = 30,
        separators: list[str] | None = None,
    ):
        self.line_tolerance = line_tolerance
        self.min_label_value_gap = min_label_value_gap
        self.separators = separators or [":", "=", " - "]

        # Padrões comuns em documentos brasileiros
        self._unit_patterns = re.compile(
            r"(mg/[dD][lL]|g/[dD][lL]|%|mm[³³]|mil/mm[³³]|U/[lL]|"
            r"mU/[lL]|ng/[dD][lL]|pg/m[lL]|nmol/[lL]|µg/[dD][lL]|"
            r"mEq/[lL]|mmHg|bpm|[kK]g|[cC]m|mm|[lL]/min|"
            r"céls?/mm[³³]|x10[³⁶⁹]/[µu]?[lL]|fL|pg)",
            re.IGNORECASE,
        )

        # Padrões de valores numéricos (incluindo intervalos de referência)
        self._value_pattern = re.compile(
            r"^[\d.,]+\s*(?:a\s*[\d.,]+)?\s*(?:" + self._unit_patterns.pattern + r")?$",
            re.IGNORECASE,
        )

        # Palavras que indicam labels/cabeçalhos
        self._label_indicators = re.compile(
            r"(resultado|exame|valor|refer[eê]ncia|material|m[eé]todo|"
            r"data|paciente|nome|idade|sexo|médico|crm|unidade|"
            r"normal|alto|baixo|positivo|negativo|reativo|"
            r"hemograma|bioquímica|hormonal|sorologia|urina)",
            re.IGNORECASE,
        )

    def parse(self, blocks: list[TextBlock]) -> list[FieldValue]:
        if not blocks: return []

        results = []

        # Estratégia 1: Separadores inline (ex: "Glicose: 95")
        inline_results = self._parse_inline_separators(blocks)
        results.extend(inline_results)

        # Estratégia 2: Proximidade espacial horizontal
        spatial_results = self._parse_spatial_proximity(blocks)
        results.extend(spatial_results)

        # Estratégia 3: Detecção de tabela
        table_results = self._parse_table_structure(blocks)
        results.extend(table_results)

        # Deduplica e prioriza resultados
        results = self._deduplicate(results)

        logger.info(f"Parser encontrou {len(results)} pares campo-valor")
        return results

    def _parse_inline_separators(self, blocks: list[TextBlock]) -> list[FieldValue]:
        results = []

        for block in blocks:
            text = block.text.strip()

            for sep in self.separators:
                if sep in text:
                    parts = text.split(sep, 1)
                    if len(parts) == 2:
                        label = parts[0].strip()
                        value = parts[1].strip()

                        # Filtrar labels muito curtos ou muito longos
                        if 2 <= len(label) <= 60 and len(value) >= 1:
                            # Não capturar horários (ex: "14:30")
                            if sep == ":" and label.isdigit():
                                continue

                            results.append(
                                FieldValue(
                                    field=label,
                                    value=value,
                                    confidence=block.confidence,
                                    source="inline_separator",
                                )
                            )
                    break  # Usar apenas o primeiro separador encontrado

        return results

    def _parse_spatial_proximity(self, blocks: list[TextBlock]) -> list[FieldValue]:
        results = []
        lines = self._group_into_lines(blocks)

        for line_blocks in lines:
            if len(line_blocks) < 2: continue

            # Ordenar blocos na linha por posição X
            line_blocks.sort(key=lambda b: b.x1)

            # Analisar pares consecutivos
            for i in range(len(line_blocks) - 1):
                left = line_blocks[i]
                right = line_blocks[i + 1]

                gap = right.x1 - left.x2

                # Verificar se há gap significativo entre os blocos
                if gap >= self.min_label_value_gap:
                    # O bloco da esquerda parece um label?
                    if self._looks_like_label(left.text) and self._looks_like_value(
                        right.text
                    ):
                        results.append(
                            FieldValue(
                                field=left.text.strip().rstrip(":").strip(),
                                value=right.text.strip(),
                                confidence=min(left.confidence, right.confidence),
                                source="spatial_proximity",
                            )
                        )

                # Caso com 3+ blocos na mesma linha (label + valor + unidade)
                if i + 2 < len(line_blocks):
                    third = line_blocks[i + 2]
                    combined_value = f"{right.text} {third.text}".strip()
                    if (
                        self._looks_like_label(left.text)
                        and self._looks_like_numeric(right.text)
                        and self._looks_like_unit(third.text)
                    ):
                        results.append(
                            FieldValue(
                                field=left.text.strip().rstrip(":").strip(),
                                value=combined_value,
                                confidence=min(
                                    left.confidence, right.confidence, third.confidence
                                ),
                                source="spatial_proximity_combined",
                            )
                        )

        return results

    def _parse_table_structure(self, blocks: list[TextBlock]) -> list[FieldValue]:
        results = []
        lines = self._group_into_lines(blocks)

        if len(lines) < 3: return results

        # Encontrar colunas por clustering das posições X dos blocos
        all_x_starts = []
        for line_blocks in lines:
            for block in line_blocks: all_x_starts.append(block.x1)

        if not all_x_starts: return results

        # Clustering simples por proximidade
        columns = self._cluster_positions(all_x_starts, tolerance=40)

        if len(columns) < 2: return results

        # Para cada linha, atribuir blocos às colunas
        for line_blocks in lines:
            col_assignments = {}
            for block in line_blocks:
                col_idx = self._find_nearest_column(block.x1, columns)
                if col_idx not in col_assignments: col_assignments[col_idx] = []
                col_assignments[col_idx].append(block)

            # Se temos conteúdo na primeira coluna e em outras
            sorted_cols = sorted(col_assignments.keys())
            if len(sorted_cols) >= 2:
                first_col = sorted_cols[0]
                second_col = sorted_cols[1]

                if first_col in col_assignments and second_col in col_assignments:
                    label_text = " ".join(b.text for b in col_assignments[first_col]).strip()
                    value_text = " ".join(b.text for b in col_assignments[second_col]).strip()

                    if (
                        self._looks_like_label(label_text)
                        and len(value_text) >= 1
                        and label_text.lower() != value_text.lower()
                    ):
                        avg_conf = sum(
                            b.confidence
                            for blocks_list in col_assignments.values()
                            for b in blocks_list
                        ) / sum(len(bl) for bl in col_assignments.values())

                        results.append(
                            FieldValue(
                                field=label_text.rstrip(":").strip(),
                                value=value_text,
                                confidence=avg_conf,
                                source="table_structure",
                            )
                        )

        return results

    def _group_into_lines(self, blocks: list[TextBlock]) -> list[list[TextBlock]]:
        if not blocks: return []

        sorted_blocks = sorted(blocks, key=lambda b: b.center_y)
        lines: list[list[TextBlock]] = []
        current_line: list[TextBlock] = [sorted_blocks[0]]

        for block in sorted_blocks[1:]:
            # Se o centro Y está próximo da linha atual
            avg_y = sum(b.center_y for b in current_line) / len(current_line)
            if abs(block.center_y - avg_y) <= self.line_tolerance:
                current_line.append(block)
            else:
                lines.append(current_line)
                current_line = [block]

        if current_line: lines.append(current_line)

        return lines

    def _cluster_positions(self, positions: list[float], tolerance: int = 40) -> list[float]:
        if not positions: return []

        sorted_pos = sorted(positions)
        clusters: list[list[float]] = [[sorted_pos[0]]]

        for pos in sorted_pos[1:]:
            if pos - clusters[-1][-1] <= tolerance:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])

        return [sum(c) / len(c) for c in clusters]

    def _find_nearest_column(self, x: float, columns: list[float]) -> int:
        min_dist = float("inf")
        min_idx = 0
        for i, col_x in enumerate(columns):
            dist = abs(x - col_x)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx

    def _looks_like_label(self, text: str) -> bool:
        text = text.strip().rstrip(":")

        if not text or len(text) < 2: return False

        # Não é um label se for apenas números
        if text.replace(".", "").replace(",", "").replace(" ", "").isdigit(): return False

        # Tem pelo menos uma letra
        if not any(c.isalpha() for c in text): return False

        # Não é muito longo (labels geralmente são curtos)
        if len(text) > 80: return False

        return True

    def _looks_like_value(self, text: str) -> bool:
        text = text.strip()
        if not text: return False

        # Valor numérico (com ou sem unidade)
        if self._looks_like_numeric(text): return True

        # Tem unidade de medida
        if self._unit_patterns.search(text): return True

        # Resultados qualitativos
        qualitative = {
            "positivo",
            "negativo",
            "reativo",
            "não reativo",
            "normal",
            "reagente",
            "não reagente",
            "presente",
            "ausente",
            "detectado",
            "não detectado",
        }
        if text.lower() in qualitative: return True

        # Qualquer texto curto pode ser um valor
        if len(text) <= 50: return True

        return False

    def _looks_like_numeric(self, text: str) -> bool:
        cleaned = text.strip().replace(",", ".").replace(" ", "")
        try:
            float(cleaned)
            return True
        except ValueError:
            pass

        # Padrão numérico com unidade ou intervalo
        if re.match(r"^\d+[.,]?\d*\s*", text): return True

        return False

    def _looks_like_unit(self, text: str) -> bool:
        return bool(self._unit_patterns.match(text.strip()))

    def _deduplicate(self, results: list[FieldValue]) -> list[FieldValue]:
        if not results: return []

        # Prioridade das fontes
        source_priority = {
            "inline_separator": 3,
            "spatial_proximity_combined": 2,
            "spatial_proximity": 2,
            "table_structure": 1,
            "vision_model": 4,
        }

        # Agrupar por campo normalizado
        groups: dict[str, list[FieldValue]] = {}
        for fv in results:
            key = fv.field.lower().strip()
            if key not in groups: groups[key] = []
            groups[key].append(fv)

        # Selecionar o melhor de cada grupo
        deduplicated = []
        for key, group in groups.items():
            # Ordenar por prioridade da fonte (desc) e confiança (desc)
            group.sort(
                key=lambda fv: (
                    source_priority.get(fv.source, 0),
                    fv.confidence,
                ),
                reverse=True,
            )
            deduplicated.append(group[0])

        return deduplicated

    def parse_from_pdf_blocks(self, pdf_blocks: list[dict]) -> list[FieldValue]:
        # Converter para TextBlock
        text_blocks = []
        for block in pdf_blocks:
            tb = TextBlock(
                text=block["text"],
                bbox=block["bbox"],
                confidence=1.0,  # Texto nativo do PDF tem confiança máxima
            )
            text_blocks.append(tb)

        return self.parse(text_blocks)

def extract_fields_from_text(raw_text: str) -> list[FieldValue]:
    results = []

    lines = raw_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line: continue

        # Padrão "Label: Valor"
        match = re.match(r"^(.{2,50?}?)\s*[:=]\s+(.+)$", line)
        if match:
            label = match.group(1).strip()
            value = match.group(2).strip()

            # Filtrar falsos positivos
            if label and value and not label[0].isdigit():
                results.append(
                    FieldValue(
                        field=label,
                        value=value,
                        confidence=0.8,
                        source="text_regex",
                    )
                )

    return results
