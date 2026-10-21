import json
import logging
import re
from pathlib import Path
import torch
from PIL import Image

logger = logging.getLogger(__name__)

class VisionDocumentExtractor:
    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        device: str | None = None,
        max_length: int = 512,
    ):
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Carregando modelo {model_name} no dispositivo {self.device}...")

        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Modelo carregado com sucesso.")

    def extract_fields(self, image: Image.Image | str | Path, prompt: str | None = None) -> dict[str, str]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            # numpy array
            image = Image.fromarray(image).convert("RGB")

        if prompt is None:
            prompt = "<s_docvqa><s_question>Extract all fields and values from this document</s_question><s_answer>"

        # Preparar input
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Preparar decoder input
        decoder_input_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)

        # Gerar output
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Decodificar output
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")

        # Tentar parsear como JSON
        fields = self._parse_output(sequence)

        logger.info(f"Modelo extraiu {len(fields)} campos")
        return fields

    def extract_with_questions(self, image: Image.Image | str | Path, questions: list[str],) -> dict[str, str]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        answers = {}

        for question in questions:
            prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            decoder_input_ids = self.processor.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids
            decoder_input_ids = decoder_input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.max_length,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
            sequence = sequence.replace(self.processor.tokenizer.pad_token, "")

            # Extrair resposta do padrão XML
            answer_match = re.search(r"<s_answer>(.*?)(?:</s_answer>|$)", sequence)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # Fallback: pegar tudo depois do prompt
                answer = sequence.replace(prompt, "").strip()

            answers[question] = answer
            logger.info(f"Q: {question} → A: {answer}")

        return answers

    def _parse_output(self, text: str) -> dict[str, str]:
        fields = {}

        # Tentar parsear tags XML-like do Donut
        # Padrão: <s_campo>valor</s_campo>
        xml_pattern = re.compile(r"<s_(\w+)>(.*?)</s_\1>")
        for match in xml_pattern.finditer(text):
            key = match.group(1).replace("_", " ")
            value = match.group(2).strip()
            if value:
                fields[key] = value

        if fields: return fields

        # Tentar parsear como JSON
        json_match = re.search(r"\{[^{}]+\}", text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass

        # Tentar parsear pares "campo: valor" no texto
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and value: fields[key] = value

        # Se nada funcionou, retornar texto raw
        if not fields:
            cleaned = text.strip()
            if cleaned: fields["_raw_output"] = cleaned

        return fields

class DonutFineTuner:
    def __init__(
        self,
        base_model: str = "naver-clova-ix/donut-base",
        output_dir: str = "./trained_model",
        device: str | None = None,
    ):
        self.base_model = base_model
        self.output_dir = output_dir

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def prepare_training_data(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
    ) -> list[dict]:
        images_dir = Path(images_dir)
        annotations_dir = Path(annotations_dir)

        data = []
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in image_extensions:
                continue

            ann_path = annotations_dir / f"{img_path.stem}.json"
            if not ann_path.exists():
                logger.warning(f"Anotação não encontrada para {img_path.name}")
                continue

            with open(ann_path, "r", encoding="utf-8") as f: annotation = json.load(f)

            fields = annotation.get("fields", {})

            # Converter para formato XML do Donut
            gt_sequence = self._fields_to_donut_format(fields)

            data.append({
                    "image_path": str(img_path),
                    "ground_truth": gt_sequence,
                    "fields": fields,
                })

        logger.info(f"Preparados {len(data)} exemplos de treino")
        return data

    def _fields_to_donut_format(self, fields: dict[str, str]) -> str:
        parts = ["<s_document>"]
        for key, value in fields.items():
            safe_key = key.replace(" ", "_").lower()
            safe_key = re.sub(r"[^a-z0-9_]", "", safe_key)
            parts.append(f"<s_{safe_key}>{value}</s_{safe_key}>")
        parts.append("</s_document>")
        return "".join(parts)

    def train(
        self,
        training_data: list[dict],
        epochs: int = 30,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        image_size: tuple[int, int] = (1280, 960),
        save_steps: int = 500,
        eval_split: float = 0.1,
    ):
        from transformers import (
            DonutProcessor,
            VisionEncoderDecoderConfig,
            VisionEncoderDecoderModel,
        )
        from torch.utils.data import Dataset, DataLoader, random_split
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Iniciando fine-tuning: {len(training_data)} amostras")
        logger.info(f"Épocas: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

        # Carregar modelo e processador
        processor = DonutProcessor.from_pretrained(self.base_model)
        model = VisionEncoderDecoderModel.from_pretrained(self.base_model)

        # Coletar tokens especiais dos ground truths
        special_tokens = set()
        for item in training_data:
            gt = item["ground_truth"]
            tokens = re.findall(r"</?s_\w+>", gt)
            special_tokens.update(tokens)

        special_tokens = sorted(special_tokens)
        if special_tokens:
            processor.tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens)})
            model.decoder.resize_token_embeddings(len(processor.tokenizer))

        # Configurar modelo
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_document>")

        # Configurar tamanho da imagem
        processor.image_processor.size = {
            "height": image_size[1],
            "width": image_size[0],
        }

        # Criar dataset
        class DocDataset(Dataset):
            def __init__(self, data, proc, max_len=512):
                self.data = data
                self.processor = proc
                self.max_len = max_len

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                image = Image.open(item["image_path"]).convert("RGB")
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

                # Tokenizar ground truth
                target = self.processor.tokenizer(
                    item["ground_truth"],
                    add_special_tokens=False,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                labels = target.input_ids.squeeze()
                # Substituir padding por -100 (ignorar no loss)
                labels[labels == self.processor.tokenizer.pad_token_id] = -100

                return {
                    "pixel_values": pixel_values,
                    "labels": labels,
                }

        dataset = DocDataset(training_data, processor, max_len=512)

        # Split treino/validação
        n_eval = max(1, int(len(dataset) * eval_split))
        n_train = len(dataset) - n_eval
        train_dataset, eval_dataset = random_split(dataset, [n_train, n_eval])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Mover modelo para device
        model.to(self.device)
        model.train()

        # Otimizador e scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # Loop de treino
        global_step = 0
        best_eval_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                if global_step % 50 == 0:
                    logger.info(
                        f"Época {epoch + 1}/{epochs}, Step {global_step}, "
                        f"Loss: {loss.item():.4f}"
                    )

                if save_steps > 0 and global_step % save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
                    logger.info(f"Checkpoint salvo: {checkpoint_dir}")

            avg_loss = epoch_loss / max(n_batches, 1)

            # Avaliação
            model.eval()
            eval_loss = 0.0
            n_eval_batches = 0
            with torch.no_grad():
                for batch in eval_loader:
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    eval_loss += outputs.loss.item()
                    n_eval_batches += 1

            avg_eval_loss = eval_loss / max(n_eval_batches, 1)

            logger.info(
                f"Época {epoch + 1}/{epochs} concluída | "
                f"Train Loss: {avg_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}"
            )

            # Salvar melhor modelo
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_dir = output_dir / "best"
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                logger.info(f"Melhor modelo salvo (loss={best_eval_loss:.4f})")

        # Salvar modelo final
        model.save_pretrained(output_dir / "final")
        processor.save_pretrained(output_dir / "final")
        logger.info(f"Treino concluído! Modelo salvo em: {output_dir}")

        return output_dir
