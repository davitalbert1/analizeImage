import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

NOMES = [
    "Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela",
    "Henrique", "Isabela", "João", "Karina", "Lucas", "Mariana",
    "Nicolas", "Olívia", "Pedro", "Rafaela", "Samuel", "Tatiane",
    "Victor", "Wagner", "Yasmin", "Zeca",
]
SOBRENOMES = [
    "Silva", "Santos", "Oliveira", "Souza", "Pereira", "Lima",
    "Ferreira", "Almeida", "Costa", "Rodrigues", "Gomes", "Martins",
    "Rocha", "Ribeiro", "Carvalho", "Mendes", "Dias", "Nunes",
    "Farias", "Barros", "Castro", "Moraes", "Teixeira", "Moreira",
]
CIDADES = [
    "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador",
    "Brasília", "Fortaleza", "Curitiba", "Manaus", "Recife",
    "Porto Alegre", "Goiânia", "Belém",
]
LABORATORIOS = [
    "Lab Central", "Clínica Saúde+", "Diagnóstico Claro",
    "Lab Saúde Total", "Exames RJ", "Centro Clínico SP",
]

def sortear_nome(rng: random.Random) -> str:
    return f"{rng.choice(NOMES)} {rng.choice(SOBRENOMES)} {rng.choice(SOBRENOMES)}"

def sortear_cpf(rng: random.Random) -> str:
    nums = [str(rng.randint(0, 9)) for _ in range(11)]
    return (
        f"{''.join(nums[:3])}.{''.join(nums[3:6])}."
        f"{''.join(nums[6:9])}-{''.join(nums[9:])}"
    )

def sortear_data(rng: random.Random, dias_atras: int = 730) -> str:
    inicial = date.today() - timedelta(days=dias_atras)
    aleatorio = inicial + timedelta(days=rng.randint(0, dias_atras))
    return aleatorio.strftime("%d/%m/%Y")

def sortear_valor_medico(rng: random.Random, minimo: int, maximo: int, unidade: str) -> str:
    return f"{rng.randint(minimo, maximo)} {unidade}"

def gerar_campos_aleatorios(rng: random.Random, indice: int) -> dict:
    return {
        "Paciente": sortear_nome(rng),
        "CPF": sortear_cpf(rng),
        "Data de Nascimento": sortear_data(rng, dias_atras=365 * 60),
        "Data do Exame": sortear_data(rng),
        "Cidade": rng.choice(CIDADES),
        "Laboratorio": rng.choice(LABORATORIOS),
        "Glicose": sortear_valor_medico(rng, 60, 200, "mg/dL"),
        "Colesterol Total": sortear_valor_medico(rng, 120, 280, "mg/dL"),
        "HDL": sortear_valor_medico(rng, 35, 80, "mg/dL"),
        "LDL": sortear_valor_medico(rng, 70, 160, "mg/dL"),
        "Triglicerídeos": sortear_valor_medico(rng, 50, 300, "mg/dL"),
        "Hemoglobina": f"{rng.uniform(11, 17):.1f} g/dL",
        "Hematócrito": f"{rng.uniform(32, 50):.1f} %",
        "Leucócitos": f"{rng.uniform(3.5, 11):.2f} /mm³",
        "Plaquetas": f"{rng.randint(140, 450)} mil/mm³",
        "Creatinina": f"{rng.uniform(0.5, 1.5):.2f} mg/dL",
        "TGO (AST)": sortear_valor_medico(rng, 10, 60, "U/L"),
        "TGP (ALT)": sortear_valor_medico(rng, 10, 60, "U/L"),
    }

def carregar_fonte(tamanho: int):
    try:
        return ImageFont.truetype("arial.ttf", tamanho)
    except Exception:
        return ImageFont.load_default()

def gerar_imagem_aleatoria(caminho: Path, campos: dict, rng: random.Random) -> None:
    largura, altura = 900, 500
    imagem = Image.new("RGB", (largura, altura), "white")
    desenho = ImageDraw.Draw(imagem)

    fonte_titulo = carregar_fonte(28)
    fonte_campo = carregar_fonte(22)

    titulos = [
        "RELATORIO DE EXAMES",
        "RESULTADOS CLINICOS",
        "LAUDO MEDICO",
        "EXAME LABORATORIAL",
    ]
    desenho.text((40, 30), rng.choice(titulos), fill="black", font=fonte_titulo)

    y = 80
    for chave, valor in campos.items():
        texto = f"{chave}: {valor}"
        desenho.text((40, y), texto, fill="black", font=fonte_campo)
        y += 38

    cor_linha = (rng.randint(0, 100), rng.randint(0, 100), rng.randint(0, 100))
    desenho.line([(40, y + 10), (largura - 40, y + 10)], fill=cor_linha, width=2)

    imagem.save(caminho, "PNG")

def gerar_itens(quantidade: int, imagens_dir: Path, anotacoes_dir: Path, seed: int | None = None) -> None:
    imagens_dir.mkdir(parents=True, exist_ok=True)
    anotacoes_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    for i in tqdm(range(1, quantidade + 1), desc="Gerando itens", unit="item"):
        nome_base = f"exame_{i:03d}"
        nome_imagem = f"{nome_base}.png"
        nome_json = f"{nome_base}.json"

        campos = gerar_campos_aleatorios(rng, i)

        gerar_imagem_aleatoria(imagens_dir / nome_imagem, campos, rng)

        anotacao = {"image": nome_imagem, "fields": campos}
        with open(anotacoes_dir / nome_json, "w", encoding="utf-8") as f:
            json.dump(anotacao, f, ensure_ascii=False, indent=2)

    print(f"✅ {quantidade} item(ns) gerado(s):")
    print(f"📁 Imagens: {imagens_dir.resolve()}")
    print(f"📁 JSONs: {anotacoes_dir.resolve()}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera itens sintéticos (imagem + JSON) para treinamento do Donut."
    )
    parser.add_argument(
        "-n", "--quantidade",
        type=int,
        default=100,
        help="Quantidade de itens a gerar (padrão: 10)",
    )
    parser.add_argument(
        "--imagens",
        default="training_data/images",
        help="Diretório de saída das imagens (padrão: training_data/images)",
    )
    parser.add_argument(
        "--anotacoes",
        default="training_data/annotations",
        help="Diretório de saída dos JSONs (padrão: training_data/annotations)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed aleatória para reproduzir os mesmos itens",
    )

    args = parser.parse_args()

    gerar_itens(
        quantidade=args.quantidade,
        imagens_dir=Path(args.imagens),
        anotacoes_dir=Path(args.anotacoes),
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
