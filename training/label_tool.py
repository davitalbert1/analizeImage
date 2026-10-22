import json
import sys
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageTk

# --- UTF-8 seguro no console Windows ---
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

class LabelTool:
    def __init__(self, images_dir: str, output_dir: str):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Coletar imagens
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        self.image_files = sorted(
            [f for f in self.images_dir.iterdir() if f.suffix.lower() in extensions]
        )

        if not self.image_files:
            print(f"❌ Nenhuma imagem encontrada em: {self.images_dir}")
            sys.exit(1)

        self.current_idx = 0
        self.fields: dict[str, str] = {}

        # Carregar anotações existentes
        self._load_existing_annotation()

        # Criar janela
        self.root = tk.Tk()
        self.root.title("DocFieldExtractor - Ferramenta de Anotação")
        self.root.geometry("1200x800")
        self._build_ui()
        self._show_current_image()

    def _build_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Painel esquerdo (imagem)
        left_frame = ttk.LabelFrame(main_frame, text="Documento")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Scrollbars para a imagem
        scrollbar_y = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x = ttk.Scrollbar(left_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Painel direito (anotações)
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Info do arquivo
        self.info_label = ttk.Label(right_frame, text="", font=("Arial", 10, "bold"))
        self.info_label.pack(pady=5)

        # Lista de campos
        fields_frame = ttk.LabelFrame(right_frame, text="Campos Anotados")
        fields_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fields_tree = ttk.Treeview(
            fields_frame, columns=("field", "value"), show="headings", height=15
        )
        self.fields_tree.heading("field", text="Campo")
        self.fields_tree.heading("value", text="Valor")
        self.fields_tree.column("field", width=150)
        self.fields_tree.column("value", width=200)
        self.fields_tree.pack(fill=tk.BOTH, expand=True)

        # Botões de edição
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="➕ Adicionar Campo", command=self._add_field).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(btn_frame, text="✏️ Editar Campo", command=self._edit_field).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(btn_frame, text="🗑️ Remover Campo", command=self._remove_field).pack(
            fill=tk.X, pady=2
        )

        # Botões de navegação
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill=tk.X, pady=10)

        ttk.Button(nav_frame, text="◀ Anterior", command=self._prev_image).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        ttk.Button(nav_frame, text="💾 Salvar", command=self._save_annotation).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        ttk.Button(nav_frame, text="Próximo ▶", command=self._next_image).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )

        # Status
        self.status_label = ttk.Label(
            right_frame, text="", foreground="green", font=("Arial", 9)
        )
        self.status_label.pack(pady=5)

        # Atalhos de teclado
        self.root.bind("<Control-s>", lambda e: self._save_annotation())
        self.root.bind("<Control-n>", lambda e: self._add_field())
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Delete>", lambda e: self._remove_field())

    def _show_current_image(self):
        if not self.image_files: return

        img_path = self.image_files[self.current_idx]
        self.info_label.config(text=f"[{self.current_idx + 1}/{len(self.image_files)}] {img_path.name}")

        # Carregar imagem
        img = Image.open(img_path)

        # Redimensionar para caber no canvas
        canvas_w = self.canvas.winfo_width() or 700
        canvas_h = self.canvas.winfo_height() or 700
        img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Carregar anotações existentes
        self._load_existing_annotation()
        self._refresh_fields_list()

    def _load_existing_annotation(self):
        if not self.image_files: return

        img_path = self.image_files[self.current_idx]
        ann_path = self.output_dir / f"{img_path.stem}.json"

        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.fields = data.get("fields", {})
        else:
            self.fields = {}

    def _refresh_fields_list(self):
        self.fields_tree.delete(*self.fields_tree.get_children())
        for field, value in self.fields.items():
            self.fields_tree.insert("", tk.END, values=(field, value))

    def _add_field(self):
        field = simpledialog.askstring("Novo Campo", "Nome do campo:", parent=self.root)
        if not field: return

        value = simpledialog.askstring("Valor do Campo", f"Valor para '{field}':", parent=self.root)
        if value is None: return

        self.fields[field] = value
        self._refresh_fields_list()
        self.status_label.config(text=f"Campo '{field}' adicionado", foreground="blue")

    def _edit_field(self):
        selection = self.fields_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um campo para editar.")
            return

        item = self.fields_tree.item(selection[0])
        old_field = item["values"][0]
        old_value = item["values"][1]

        new_value = simpledialog.askstring(
            "Editar Valor",
            f"Novo valor para '{old_field}':",
            initialvalue=old_value,
            parent=self.root,
        )
        if new_value is not None:
            self.fields[old_field] = new_value
            self._refresh_fields_list()
            self.status_label.config(
                text=f"Campo '{old_field}' atualizado", foreground="blue"
            )

    def _remove_field(self):
        selection = self.fields_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um campo para remover.")
            return

        item = self.fields_tree.item(selection[0])
        field = item["values"][0]

        if messagebox.askyesno("Confirmar", f"Remover campo '{field}'?"):
            del self.fields[field]
            self._refresh_fields_list()
            self.status_label.config(text=f"Campo '{field}' removido", foreground="red")

    def _save_annotation(self):
        if not self.image_files: return

        img_path = self.image_files[self.current_idx]
        ann_path = self.output_dir / f"{img_path.stem}.json"

        data = {"image": img_path.name, "fields": self.fields,}

        with open(ann_path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

        self.status_label.config(text=f"✅ Salvo: {ann_path.name}", foreground="green")

    def _prev_image(self):
        if self.current_idx > 0:
            self._save_annotation()  # Auto-save
            self.current_idx -= 1
            self._show_current_image()

    def _next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self._save_annotation()  # Auto-save
            self.current_idx += 1
            self._show_current_image()

    def run(self):
        print(f"🏷️ Ferramenta de anotação iniciada")
        print(f"   Imagens: {self.images_dir} ({len(self.image_files)} arquivos)")
        print(f"   Saída: {self.output_dir}")
        print(f"   Atalhos: Ctrl+S=Salvar, Ctrl+N=Novo campo, Setas=Navegar")
        self.root.mainloop()

def run_label_tool(images_dir: str, output_dir: str):
    tool = LabelTool(images_dir=images_dir, output_dir=output_dir)
    tool.run()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ferramenta de anotação de documentos")
    parser.add_argument("--images", required=True, help="Diretório com imagens")
    parser.add_argument(
        "--output",
        default="./training_data/annotations",
        help="Diretório de saída",
    )

    args = parser.parse_args()
    run_label_tool(args.images, args.output)
