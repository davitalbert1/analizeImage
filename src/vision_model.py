import os
import json
import logging
import re
import random
import gc
import shutil
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from src.utils import get_optimized_training_config, print_resource_report

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

        return {"pixel_values": pixel_values, "labels": labels}

class DonutFineTuner:
    def __init__(
        self,
        base_model: str = "naver-clova-ix/donut-base",
        output_dir: str = "./trained_model",
        device: str | None = None,
        max_length: int | None = None,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.max_length = max_length  # None = calcular dinamicamente nos dados

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def prepare_training_data(self, images_dir: str | Path, annotations_dir: str | Path) -> list[dict]:
        images_dir = Path(images_dir)
        annotations_dir = Path(annotations_dir)

        data = []
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in image_extensions: continue

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

    def _limpar_modelos_antigos(self, output_dir: Path) -> None:
        # Mantém apenas 1 modelo na raiz (sobrescrito a cada salva): remove subpastas antigas
        for sub in ("best", "final", "last_epoch"):
            d = output_dir / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"Diretório de modelo antigo removido: {d}")
        for pattern in ("epoch_*", "checkpoint-*"):
            for d in output_dir.glob(pattern):
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"Diretório de modelo antigo removido: {d}")

    def train(
        self,
        training_data: list[dict],
        epochs: int = 5,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        max_steps_per_epoch: int | None = 500,
        warmup_steps: int = 100,
        image_size: tuple[int, int] = (1280, 960),
        save_steps: int = 500,
        eval_split: float = 0.1,
        seed: int | None = None,
        mixed_precision: bool = True,
        ema_decay: float = 0.0,
        auto_tune: bool = True,
        gradient_accumulation_steps: int = 4,
        gradient_checkpointing: bool = True,
    ):
        from transformers import (
            DonutProcessor,
            VisionEncoderDecoderModel,
        )
        from torch.utils.data import DataLoader, random_split
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Seed para reprodutibilidade
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            logger.info(f"Seed definido: {seed}")

        # Auto-tuning de recursos
        if auto_tune:
            tune_cfg = get_optimized_training_config(current_batch_size=batch_size, current_workers=0)
            print_resource_report(tune_cfg)
            batch_size = tune_cfg["recommended_batch_size"]
            num_workers = tune_cfg["recommended_workers"]
            if mixed_precision and tune_cfg["precision_str"] == "float32":
                mixed_precision = False
            logger.info(f"Auto-tune: batch_size={batch_size}, workers={num_workers}, amp={mixed_precision}")
        else:
            num_workers = 0

        logger.info(f"Iniciando fine-tuning: {len(training_data)} amostras")
        logger.info(f"Épocas: {epochs}, Batch: {batch_size}, LR: {learning_rate}, AMP: {mixed_precision}")

        # Carregar modelo e processador com retry
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                processor = DonutProcessor.from_pretrained(self.base_model)
                model = VisionEncoderDecoderModel.from_pretrained(self.base_model)
                break
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Falha ao carregar modelo após {max_retries} tentativas: {e}")
                    raise
                logger.warning(f"Tentativa {attempt}/{max_retries} falhou ao carregar modelo: {e}. Tentando novamente...")
                import time
                time.sleep(2 ** attempt)

        # Coletar tokens especiais dos ground truths
        special_tokens = set()
        all_tags = set()
        for item in training_data:
            gt = item["ground_truth"]
            tokens = re.findall(r"</?s_\w+>", gt)
            special_tokens.update(tokens)
            # Extrair nomes de tags (sem os colchetes e barra)
            tags = re.findall(r"<s_(\w+)>", gt)
            all_tags.update(tags)

        special_tokens = sorted(special_tokens)
        if special_tokens:
            processor.tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens)})
            model.decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)

        # Calcular comprimento máximo de sequência dinamicamente a partir dos ground truths
        if self.max_length is None:
            max_gt_len = max(
                len(processor.tokenizer(item["ground_truth"], add_special_tokens=False).input_ids)
                for item in training_data
            )
            # Adicionar margem de segurança e alinhar para múltiplo de 8
            self.max_length = min(1024, max(max_gt_len + 32, 128))
            self.max_length = ((self.max_length + 7) // 8) * 8
            logger.info(f"Comprimento máximo de sequência calculado dinamicamente: {self.max_length}")
        else:
            logger.info(f"Usando comprimento de sequência especificado: {self.max_length}")

        # Salvar vocabulário de tags para consistência entre treino e inferência
        tags_path = output_dir / "tags.json"
        tag_list = sorted(all_tags)
        with open(tags_path, "w", encoding="utf-8") as f:
            json.dump({"tags": tag_list, "num_tags": len(tag_list)}, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulário de tags salvo: {len(tag_list)} tags em {tags_path}")

        # Inicializar arquivo de métricas JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"epochs": []}, f, ensure_ascii=False, indent=2)

        # Configurar modelo
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_document>")

        # Gradient checkpointing economiza muita VRAM na Swin Transformer e BART decoder
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            # Desativar use_cache para evitar warning do transformers:
            # "use_cache=True is incompatible with gradient checkpointing"
            model.config.use_cache = False
            logger.info("Gradient checkpointing ativado (redução drástica de VRAM)")

        # Configurar tamanho da imagem
        processor.image_processor.size = {"height": image_size[1], "width": image_size[0]}

        # Criar dataset com comprimento de sequência dinâmico
        dataset = DocDataset(training_data, processor, max_len=self.max_length)

        # Split treino/validação
        n_eval = max(1, int(len(dataset) * eval_split))
        n_train = len(dataset) - n_eval
        train_dataset, eval_dataset = random_split(dataset, [n_train, n_eval])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Mover modelo para device
        model.to(self.device)
        model.train()

        from transformers import VisionEncoderDecoderModel

        # Otimizador e scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # EMA (Exponential Moving Average)
        ema_model = None
        if ema_decay > 0:
            import copy
            ema_model = copy.deepcopy(model)
            ema_model.eval()
            logger.info(f"EMA ativado com decay={ema_decay}")

        # Mixed precision (AMP)
        scaler = torch.amp.GradScaler("cuda") if (mixed_precision and torch.cuda.is_available()) else None
        use_amp = scaler is not None
        if use_amp:
            logger.info("Precisão mista (AMP) ativada")

        # Loop de treino
        global_step = 0
        best_eval_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Épocas", unit="época"):
            # Carregar modelo da época anterior (se não for a primeira)
            # O modelo é mantido em 1 único arquivo na raiz (sobrescrito a cada salva)
            if epoch > 0:
                single_model_path = output_dir / "model.safetensors"
                if single_model_path.exists():
                    logger.info(f"Carregando modelo da época {epoch}...")
                    # Load checkpoint weights in-place (NOT by replacing the model
                    # object) to preserve parameter-tensor identity.  The optimizer,
                    # scheduler, GradScaler and EMA model all hold references to the
                    # *same* parameter tensors that were used when they were
                    # created.  If we did  model = ...from_pretrained(...)  we would
                    # get a new model whose parameters the optimizer/scaler do not
                    # know about, which causes:
                    # "AssertionError: No inf checks were recorded for this
                    #  optimizer."
                    # load_state_dict copies data into the existing tensors via
                    # copy_() so tensor identity is retained and the optimizer /
                    # scaler continue to work.
                    checkpoint_model = VisionEncoderDecoderModel.from_pretrained(output_dir)
                    model.load_state_dict(checkpoint_model.state_dict())
                    del checkpoint_model  # free the temporary model immediately
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    model.to(self.device)
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            # Ajustar total do progress bar quando há limite de steps por época
            pbar_total = len(train_loader)
            if max_steps_per_epoch is not None and max_steps_per_epoch > 0:
                pbar_total = min(pbar_total, max_steps_per_epoch)

            pbar = tqdm(train_loader, desc=f"Época {epoch + 1}/{epochs}", unit="batch", leave=False, total=pbar_total)
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(pbar):
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Forward pass com precisão mista (AMP) quando disponível
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    # Normaliza loss para gradient accumulation
                    loss_scaled = loss / max(gradient_accumulation_steps, 1)

                # Proteção contra NaN/Inf no loss
                if not torch.isfinite(loss):
                    logger.warning(f"Loss não-finito detectado no step {global_step}, pulando batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if use_amp:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                # Atualiza pesos a cada gradient_accumulation_steps ou no último batch
                is_accum_step = ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))
                if is_accum_step:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        # scaler.step(optimizer) may skip the optimizer step when
                        # inf/NaN gradients are detected after unscaling. We
                        # compare the loss scale before and after the step to only
                        # call scheduler.step() when the optimizer was actually
                        # stepped, avoiding the "lr_scheduler.step() before
                        # optimizer.step()" warning.
                        # (The old optimizer._step_count attribute was removed in
                        #  modern PyTorch, hence this scale-based check.)
                        prev_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        # If the scale is unchanged, the optimizer step was not
                        # skipped by the scaler (no inf/NaN gradients).
                        if scaler.get_scale() == prev_scale:
                            scheduler.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                    # Atualizar EMA
                    if ema_model is not None:
                        with torch.no_grad():
                            for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)

                # Acurácia de token-level (ignorando padding -100)
                with torch.no_grad():
                    logits = outputs.logits
                    predictions = logits.argmax(dim=-1)
                    mask = labels != -100
                    if mask.sum() > 0:
                        correct = (predictions == labels) & mask
                        batch_acc = correct.sum().float() / mask.sum().float()
                    else:
                        batch_acc = torch.tensor(0.0, device=self.device)

                epoch_loss += loss.item()
                epoch_acc += batch_acc.item()
                n_batches += 1
                global_step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc.item():.4f}"})

                # Limite máximo de steps por época (quando definido)
                if max_steps_per_epoch is not None and max_steps_per_epoch > 0 and batch_idx + 1 >= max_steps_per_epoch:
                    logger.info(f"Limite de {max_steps_per_epoch} steps por época atingido. Encerrando época {epoch + 1} antecipadamente.")
                    break

                if global_step % 50 == 0:
                    logger.info(f"Época {epoch + 1}/{epochs}, Step {global_step}, Loss: {loss.item():.4f}")

                if save_steps > 0 and global_step % save_steps == 0:
                    # Remover checkpoints antigos mantendo apenas o mais recente
                    for old_cp in output_dir.glob("checkpoint-*"): shutil.rmtree(old_cp, ignore_errors=True)
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
                    logger.info(f"Checkpoint salvo: {checkpoint_dir}")

            pbar.close()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Limpar memória GPU após época de treino antes da avaliação
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

            # Avaliação
            model.eval()
            eval_loss = 0.0
            eval_acc = 0.0
            n_eval_batches = 0
            # Avaliar apenas a quantidade de steps executados nesta época
            # (respeita max_steps_per_epoch), em vez do dataset inteiro.
            eval_steps = min(len(eval_loader), max(n_batches, 1))
            pbar_eval = tqdm(eval_loader, desc="Avaliação", unit="batch", leave=False, total=eval_steps)
            with torch.no_grad():
                for batch_idx, batch in enumerate(pbar_eval):
                    pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                    labels = batch["labels"].to(self.device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(pixel_values=pixel_values, labels=labels)
                    eval_loss += outputs.loss.item()

                    # Acurácia de token-level na avaliação
                    with torch.no_grad():
                        logits = outputs.logits
                        predictions = logits.argmax(dim=-1)
                        mask = labels != -100
                        if mask.sum() > 0:
                            correct = (predictions == labels) & mask
                            batch_acc = correct.sum().float() / mask.sum().float()
                        else:
                            batch_acc = torch.tensor(0.0, device=self.device)
                    eval_acc += batch_acc.item()

                    n_eval_batches += 1
                    pbar_eval.set_postfix({"eval_loss": f"{outputs.loss.item():.4f}"})

                    if batch_idx + 1 >= eval_steps:
                        break
            pbar_eval.close()

            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

            avg_eval_loss = eval_loss / max(n_eval_batches, 1)
            avg_train_acc = epoch_acc / max(n_batches, 1)
            avg_eval_acc = eval_acc / max(n_eval_batches, 1)

            logger.info(
                f"Época {epoch + 1}/{epochs} concluída | "
                f"Train Loss: {avg_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | "
                f"Train Acc: {avg_train_acc:.4f} | Eval Acc: {avg_eval_acc:.4f}"
            )

            # Salvar métricas consolidadas em JSON (metrics.json no diretório raiz do modelo)
            metrics_path = output_dir / "metrics.json"
            try:
                with open(metrics_path, "r", encoding="utf-8") as f: metrics = json.load(f)
            except Exception:
                metrics = {"epochs": []}
            metrics["epochs"].append({
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 6),
                "eval_loss": round(avg_eval_loss, 6),
                "train_acc": round(avg_train_acc, 6),
                "eval_acc": round(avg_eval_acc, 6),
            })
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            # Salvar o único modelo na raiz, sobrescrevendo o anterior (usar pesos EMA se disponível)
            save_model = ema_model if ema_model is not None else model
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                logger.info(f"Nova melhor perda de validação: {best_eval_loss:.4f}")
            save_model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            logger.info(f"Modelo da época {epoch + 1} salvo em: {output_dir} (sobrescrito)")
            # Apagar subpastas/modelos antigos, mantendo apenas o modelo único na raiz
            self._limpar_modelos_antigos(output_dir)

        # Salvar o único modelo final na raiz, sobrescrevendo (usar pesos EMA se disponível)
        save_model = ema_model if ema_model is not None else model
        save_model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        # Apagar subpastas antigas, mantendo apenas o modelo único na raiz
        self._limpar_modelos_antigos(output_dir)
        logger.info(f"Treino concluído! Modelo único salvo (sobrescrito) em: {output_dir}")

        return output_dir
