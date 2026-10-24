import argparse
import json
import random
import string
import sys
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from tqdm import tqdm

try:
    import numpy as np
except ImportError:  # numpy é opcional (usado apenas em ruído/recorte)
    np = None

# UTF-8 seguro no console Windows
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

ESTADOS = [
    "SP", "RJ", "MG", "BA", "DF", "CE", "PR", "AM", "PE", "RS", "GO", "PA",
]

# Vocabulário neutro reutilizável em qualquer domínio
GENERICO_TERMOS = [
    "Alpha", "Beta", "Delta", "Omega", "Nova", "Aurora", "Atlas",
    "Bravo", "Cosmos", "Eureka", "Fenix", "Horizonte", "Íris", "Lumen",
    "Nébula", "Orion", "Prisma", "Quantum", "Rubi", "Solar", "Titan",
    "Ultra", "Vertex", "Zenite",
]

CATEGORIAS = [
    "Serviço", "Produto", "Contrato", "Pedido", "Chamado", "Projeto",
    "Cadastro", "Solicitação", "Registro", "Ordem",
]

STATUS = [
    "Aberto", "Fechado", "Pendente", "Em análise", "Aprovado",
    "Concluído", "Cancelado", "Suspenso",
]

# Nomes de campos universais reutilizados em todos os tipos de documento.
CAMPOS_COMUNS = {
    "Código": lambda rng: f"{rng.choice(string.ascii_uppercase)}{rng.randint(1000, 9999)}-{rng.randint(10, 99)}",
    "Data": lambda rng: sortear_data(rng),
    "Nome": lambda rng: sortear_nome(rng),
    "Identificador": lambda rng: "".join(rng.choices(string.digits, k=9)),
    "Categoria": lambda rng: rng.choice(CATEGORIAS),
    "Status": lambda rng: rng.choice(STATUS),
    "Responsável": lambda rng: sortear_nome(rng),
    "Observação": lambda rng: rng.choice([
        "Sem observações adicionais.",
        "Verificar pendências.",
        "Processado conforme solicitação.",
        "Aguardando aprovação.",
        "Documento conferido.",
        "Sujeito a conferência posterior.",
    ]),
    "Valor": lambda rng: f"{rng.uniform(10, 9999):.2f}",
    "Quantidade": lambda rng: str(rng.randint(1, 500)),
    "Cidade": lambda rng: rng.choice(CIDADES),
    "Estado": lambda rng: rng.choice(ESTADOS),
    "Referência": lambda rng: f"REF-{rng.randint(10000, 99999)}",
}

# Perfis de documento (tipos genéricos)
# Cada perfil define: título, subtítulo, pools de rótulos (fields) e o
# gerador de valor de cada rótulo. Os rótulos são reaproveitados entre
# perfis para manter o vocabulário de tags do tokenizer pequeno e estável.
TIPOS_DOCUMENTO: dict[str, dict] = {
    "financeiro": {
        "titulos": [
            "RELATÓRIO FINANCEIRO", "COMPROVANTE DE PAGAMENTO",
            "EXTRATO DE CONTA", "DEMONSTRATIVO DE VALORES",
            "NOTA DE DÉBITO", "CONTROLE DE PAGAMENTOS",
        ],
        "subtitulos": ["Período de referência", "Consolidado", "Detalhado"],
        "fields": {
            "Descrição": lambda rng: rng.choice([
                "Transferência entre contas", "Pagamento de fornecedor",
                "Crédito de operação", "Ajuste de saldo",
                "Despesa administrativa", "Repasse de valores",
            ]),
            "Valor": lambda rng: f"R$ {rng.uniform(10, 50000):.2f}",
            "Moeda": lambda rng: "BRL",
            "Centro de Custo": lambda rng: rng.choice(GENERICO_TERMOS),
            "Conta": lambda rng: f"{rng.randint(1000, 9999)}-{rng.randint(10, 99)}",
            "Banco": lambda rng: rng.choice(GENERICO_TERMOS),
            "Tipo de Operação": lambda rng: rng.choice([
                "Crédito", "Débito", "Transferência", "Pagamento",
            ]),
            "Data de Vencimento": lambda rng: sortear_data(rng),
            "Valor Total": lambda rng: f"R$ {rng.uniform(100, 100000):.2f}",
            "Pago": lambda rng: rng.choice(["Sim", "Não", "Parcial"]),
        },
    },
    "transporte": {
        "titulos": [
            "COMPROVANTE DE TRANSPORTE", "NOTA DE REMESSA",
            "RELATÓRIO DE ROTA", "TERMO DE ENTREGA",
            "REGISTRO DE DESPACHO", "ACOMPANHAMENTO DE CARGA",
        ],
        "subtitulos": ["Origem", "Destino", "Rota"],
        "fields": {
            "Origem": lambda rng: rng.choice(CIDADES),
            "Destino": lambda rng: rng.choice(CIDADES),
            "Motorista": lambda rng: sortear_nome(rng),
            "Placa": lambda rng: f"{''.join(rng.choices(string.ascii_uppercase, k=3))}{rng.randint(0, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)}",
            "Distância": lambda rng: f"{rng.randint(5, 1200)} km",
            "Peso": lambda rng: f"{rng.uniform(1, 5000):.1f} kg",
            "Data de Coleta": lambda rng: sortear_data(rng),
            "Data de Entrega": lambda rng: sortear_data(rng),
            "Tipo de Frete": lambda rng: rng.choice([
                "Normal", "Expresso", "Programado", "Especial",
            ]),
            "Cidade": lambda rng: rng.choice(CIDADES),
        },
    },
    "recursos_humanos": {
        "titulos": [
            "REGISTRO DE COLABORADOR", "FORMULÁRIO DE ADMISSÃO",
            "SOLICITAÇÃO DE FÉRIAS", "COMPROVANTE DE TREINAMENTO",
            "FICHA DE CADASTRO", "AVALIAÇÃO DE DESEMPENHO",
        ],
        "subtitulos": ["Colaborador", "Cadastro", "Movimentação"],
        "fields": {
            "Colaborador": lambda rng: sortear_nome(rng),
            "Cargo": lambda rng: rng.choice([
                "Analista", "Técnico", "Coordenador", "Assistente",
                "Gerente", "Especialista", "Operacional",
            ]),
            "Departamento": lambda rng: rng.choice(GENERICO_TERMOS),
            "Data de Admissão": lambda rng: sortear_data(rng),
            "Tipo de Contrato": lambda rng: rng.choice([
                "CLT", "PJ", "Estágio", "Temporário",
            ]),
            "Carga Horária": lambda rng: rng.choice([
                "40h semanais", "44h semanais", "30h semanais", "Escala 12x36",
            ]),
            "Status": lambda rng: rng.choice(STATUS),
            "Observação": lambda rng: rng.choice([
                "Sem observações adicionais.",
                "Documentação pendente.",
                "Processado conforme solicitação.",
                "Aguardando aprovação.",
            ]),
            "Identificador": lambda rng: "".join(rng.choices(string.digits, k=9)),
        },
    },
    "saude": {
        "titulos": [
            "RELATÓRIO DE EXAMES", "RESULTADOS CLÍNICOS",
            "LAUDO MÉDICO", "EXAME LABORATORIAL",
            "COMPROVANTE DE ATENDIMENTO", "FICHA DE ACOMPANHAMENTO",
        ],
        "subtitulos": ["Paciente", "Atendimento", "Laudo"],
        "fields": {
            "Paciente": lambda rng: sortear_nome(rng),
            "Profissional": lambda rng: sortear_nome(rng),
            "Data do Exame": lambda rng: sortear_data(rng),
            "Tipo de Exame": lambda rng: rng.choice([
                "Laboratorial", "Imagem", "Clínico", "Funcional",
            ]),
            "Unidade": lambda rng: rng.choice(GENERICO_TERMOS),
            "Resultado": lambda rng: rng.choice([
                "Dentro do esperado", "Alterado", "Em observação",
                "Normal", "Acompanhamento necessário",
            ]),
            "Observação": lambda rng: rng.choice([
                "Sem observações adicionais.",
                "Repetir avaliação em 90 dias.",
                "Encaminhado para acompanhamento.",
                "Documento conferido.",
            ]),
            "Valor": lambda rng: f"{rng.uniform(10, 900):.2f}",
        },
    },
    "logistica": {
        "titulos": [
            "NOTA DE EMBARQUE", "CONTROLE DE ARMAZÉM",
            "INVENTÁRIO DE ESTOQUE", "REGISTRO DE EXPEDIÇÃO",
            "PEDIDO DE COMPRA", "CONFERÊNCIA DE CARGA",
        ],
        "subtitulos": ["Armazém", "Expedição", "Estoque"],
        "fields": {
            "Produto": lambda rng: f"{rng.choice(GENERICO_TERMOS)} {rng.randint(1, 999)}",
            "SKU": lambda rng: "".join(rng.choices(string.ascii_uppercase + string.digits, k=8)),
            "Quantidade": lambda rng: str(rng.randint(1, 5000)),
            "Armazém": lambda rng: rng.choice(GENERICO_TERMOS),
            "Setor": lambda rng: f"{rng.choice(string.ascii_uppercase)}{rng.randint(1, 20)}",
            "Data de Entrada": lambda rng: sortear_data(rng),
            "Data de Saída": lambda rng: sortear_data(rng),
            "Fornecedor": lambda rng: sortear_nome(rng),
            "Valor Unitário": lambda rng: f"R$ {rng.uniform(1, 5000):.2f}",
            "Status": lambda rng: rng.choice(STATUS),
        },
    },
    "energia": {
        "titulos": [
            "FATURA DE ENERGIA", "RELATÓRIO DE CONSUMO",
            "MEDIÇÃO DE UNIDADE", "DETALHAMENTO DE TARIFA",
            "HISTÓRICO DE CONSUMO", "COMPROVANTE DE LEITURA",
        ],
        "subtitulos": ["Consumo", "Tarifa", "Leitura"],
        "fields": {
            "Unidade Consumidora": lambda rng: str(rng.randint(10000000, 99999999)),
            "Consumo": lambda rng: f"{rng.randint(20, 5000)} kWh",
            "Leitura Anterior": lambda rng: str(rng.randint(1000, 99999)),
            "Leitura Atual": lambda rng: str(rng.randint(10000, 199999)),
            "Tarifa": lambda rng: rng.choice(["B1", "B2", "B3", "A1", "A2", "G1"]),
            "Vencimento": lambda rng: sortear_data(rng),
            "Valor": lambda rng: f"R$ {rng.uniform(50, 3000):.2f}",
            "Bandeira": lambda rng: rng.choice(["Verde", "Amarela", "Vermelha"]),
        },
    },
    "telecom": {
        "titulos": [
            "FATURA DE SERVIÇO", "RELATÓRIO DE ASSINATURA",
            "DETALHAMENTO DE USO", "COMPROVANTE DE INSTALAÇÃO",
            "SOLICITAÇÃO DE PLANO", "CONTROLE DE CONTRATO",
        ],
        "subtitulos": ["Assinante", "Plano", "Uso"],
        "fields": {
            "Assinante": lambda rng: sortear_nome(rng),
            "Linha": lambda rng: f"({rng.randint(11, 99)}) {rng.randint(90000, 99999)}-{rng.randint(1000, 9999)}",
            "Plano": lambda rng: rng.choice(GENERICO_TERMOS),
            "Data de Ativação": lambda rng: sortear_data(rng),
            "Valor da Mensalidade": lambda rng: f"R$ {rng.uniform(29, 300):.2f}",
            "Data de Vencimento": lambda rng: sortear_data(rng),
            "Status": lambda rng: rng.choice(STATUS),
            "Observação": lambda rng: rng.choice([
                "Sem observações adicionais.",
                "Ajuste de titularidade solicitado.",
                "Aguardando instalação.",
                "Documento conferido.",
            ]),
        },
    },
    "seguranca": {
        "titulos": [
            "RELATÓRIO DE OCORRÊNCIA", "REGISTRO DE OCORRÊNCIA",
            "COMPROVANTE DE APÓLICE", "RELATÓRIO DE PATRULHA",
            "REGISTRO DE VISITA", "CONTROLE DE RISCO",
        ],
        "subtitulos": ["Ocorrência", "Apólice", "Vistoria"],
        "fields": {
            "Segurado": lambda rng: sortear_nome(rng),
            "Número da Apólice": lambda rng: f"{rng.randint(100000, 999999)}",
            "Tipo de Cobertura": lambda rng: rng.choice(GENERICO_TERMOS),
            "Vigência": lambda rng: f"{sortear_data(rng)} a {sortear_data(rng)}",
            "Valor da Apólice": lambda rng: f"R$ {rng.uniform(100, 20000):.2f}",
            "Ocorrido em": lambda rng: sortear_data(rng),
            "Cidade": lambda rng: rng.choice(CIDADES),
            "Estado": lambda rng: rng.choice(ESTADOS),
            "Status": lambda rng: rng.choice(STATUS),
        },
    },
    "academico": {
        "titulos": [
            "DECLARAÇÃO DE MATRÍCULA", "COMPROVANTE DE NOTAS",
            "DECLARAÇÃO DE FREQUÊNCIA", "HISTÓRICO ESCOLAR",
            "CERTIFICADO DE CONCLUSÃO", "FICHA DE ALUNO",
        ],
        "subtitulos": ["Aluno", "Disciplina", "Período"],
        "fields": {
            "Aluno": lambda rng: sortear_nome(rng),
            "Matrícula": lambda rng: "".join(rng.choices(string.digits, k=8)),
            "Curso": lambda rng: rng.choice(GENERICO_TERMOS),
            "Período": lambda rng: f"{rng.randint(2018, 2026)}.{rng.randint(1, 2)}",
            "Disciplina": lambda rng: rng.choice(GENERICO_TERMOS),
            "Nota": lambda rng: f"{rng.uniform(0, 10):.1f}",
            "Frequência": lambda rng: f"{rng.randint(60, 100)}%",
            "Situação": lambda rng: rng.choice(["Aprovado", "Reprovado", "Em andamento"]),
            "Data de Emissão": lambda rng: sortear_data(rng),
        },
    },
    "generico": {
        "titulos": [
            "DOCUMENTO", "REGISTRO GERAL", "FORMULÁRIO",
            "COMPROVANTE", "RELATÓRIO", "FICHA DE REGISTRO",
        ],
        "subtitulos": ["Registro", "Geral", "Controle"],
        "fields": dict(CAMPOS_COMUNS),
    },
}

# Helpers de sorteio de valores
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
    aleatorio = inicial + timedelta(days=rng.randint(0, max(dias_atras, 0)))
    return aleatorio.strftime("%d/%m/%Y")

ESTILOS_VISUAIS = [
    {"fonte": "arial.ttf", "cor_texto": (0, 0, 0), "cor_fundo": (255, 255, 255), "fundo_cinza": False},
    {"fonte": "arialbd.ttf", "cor_texto": (20, 20, 60), "cor_fundo": (255, 255, 255), "fundo_cinza": False},
    {"fonte": "calibri.ttf", "cor_texto": (30, 30, 30), "cor_fundo": (245, 245, 240), "fundo_cinza": True},
    {"fonte": "times.ttf", "cor_texto": (0, 0, 0), "cor_fundo": (255, 255, 255), "fundo_cinza": False},
    {"fonte": "verdana.ttf", "cor_texto": (40, 40, 40), "cor_fundo": (250, 250, 250), "fundo_cinza": True},
    {"fonte": "consola.ttf", "cor_texto": (0, 40, 0), "cor_fundo": (255, 255, 255), "fundo_cinza": False},
    {"fonte": "cour.ttf", "cor_texto": (60, 0, 0), "cor_fundo": (255, 252, 240), "fundo_cinza": False},
]

# Fontes de fallback caso a fonte preferida não exista no sistema
FONTES_FALLBACK = [
    "arial.ttf", "calibri.ttf", "segoeui.ttf", "times.ttf",
    "verdana.ttf", "DejaVuSans.ttf", "DejaVuSansMono.ttf",
]

CORES_TITULO = [
    (0, 0, 0), (20, 40, 90), (90, 20, 20), (25, 70, 40), (60, 30, 80),
]

# Layouts possíveis
LAYOUTS = ["lista", "lista_alternada", "duas_colunas", "tabela", "blocos"]

def gerar_campos_aleatorios(rng: random.Random, tipo_doc: str | None = None) -> dict:
    if tipo_doc is None: tipo_doc = rng.choice(list(TIPOS_DOCUMENTO.keys()))

    perfil = TIPOS_DOCUMENTO[tipo_doc]
    pool = dict(perfil["fields"])

    # Campos comuns garantidos (não duplicados com os do perfil)
    for chave in ("Código", "Data", "Nome", "Identificador"):
        if chave not in pool: pool[chave] = CAMPOS_COMUNS[chave]

    # Garante um identificador-like único quando existir "Identificador"
    n_campos = rng.randint(6, min(11, len(pool)))
    chaves = rng.sample(list(pool.keys()), n_campos)

    # Mantém "Código" e "Data" presentes na maioria dos documentos
    for obrigatorio in ("Código", "Data"):
        if obrigatorio in pool and obrigatorio not in chaves and rng.random() < 0.75:
            chaves[0] = obrigatorio

    campos: dict[str, str] = {}
    for chave in chaves:
        try:
            valor = pool[chave](rng)
        except Exception:
            valor = ""
        if valor is None: valor = ""
        campos[chave] = str(valor)

    return campos

_cache_fontes: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

def carregar_fonte(tamanho: int, nome_fonte: str = "arial.ttf"):
    chave = (nome_fonte, tamanho)
    if chave in _cache_fontes: return _cache_fontes[chave]

    candidatos = [nome_fonte] + [f for f in FONTES_FALLBACK if f != nome_fonte]
    fonte = None
    for candidato in candidatos:
        try:
            fonte = ImageFont.truetype(candidato, tamanho)
            break
        except Exception:
            continue

    if fonte is None: fonte = ImageFont.load_default()

    _cache_fontes[chave] = fonte
    return fonte

def _largura_texto(desenho: ImageDraw.ImageDraw, texto: str, fonte) -> int:
    try:
        return int(desenho.textlength(texto, font=fonte))
    except Exception:
        return int(len(texto) * 8)

def _render_lista(
    desenho: ImageDraw.ImageDraw,
    campos: dict,
    x: int,
    y: int,
    largura_util: int,
    fonte_campo,
    cor_texto,
    cor_linha,
    altura_linha: int,
    registro: dict | None = None,
) -> int:
    for chave, valor in campos.items():
        rotulo = f"{chave}:"
        desenho.text((x, y), rotulo, fill=cor_texto, font=fonte_campo)
        largura_rotulo = _largura_texto(desenho, rotulo, fonte_campo)
        if valor:
            valor_txt = str(valor)
            largura_valor = _largura_texto(desenho, valor_txt, fonte_campo)
            while largura_valor > largura_util - largura_rotulo - 12 and len(valor_txt) > 4:
                valor_txt = valor_txt[:-2]
                largura_valor = _largura_texto(desenho, valor_txt, fonte_campo)
            desenho.text((x + largura_rotulo + 8, y), valor_txt, fill=cor_texto, font=fonte_campo)
            if registro is not None: registro[chave] = valor_txt
        y += altura_linha
    return y

def _render_lista_alternada(
    desenho: ImageDraw.ImageDraw,
    campos: dict,
    x: int,
    y: int,
    largura_util: int,
    fonte_campo,
    cor_texto,
    cor_linha,
    altura_linha: int,
    cor_fundo_alternado,
    registro: dict | None = None,
) -> int:
    pos_col = 0
    for chave, valor in campos.items():
        if pos_col % 2 == 1:
            caixa = [x, y - 4, x + largura_util, y + altura_linha - 6]
            desenho.rectangle(caixa, fill=cor_fundo_alternado)

        rotulo = f"{chave}:"
        desenho.text((x, y), rotulo, fill=cor_texto, font=fonte_campo)
        largura_rotulo = _largura_texto(desenho, rotulo, fonte_campo)
        if valor:
            valor_txt = str(valor)
            largura_valor = _largura_texto(desenho, valor_txt, fonte_campo)
            while largura_valor > largura_util - largura_rotulo - 20 and len(valor_txt) > 4:
                valor_txt = valor_txt[:-2]
                largura_valor = _largura_texto(desenho, valor_txt, fonte_campo)
            desenho.text((x + largura_rotulo + 8, y), valor_txt, fill=cor_texto, font=fonte_campo)
            if registro is not None: registro[chave] = valor_txt
        y += altura_linha
        pos_col += 1
    return y

def _render_blocos(
    desenho: ImageDraw.ImageDraw,
    campos: dict,
    x: int,
    y: int,
    largura_util: int,
    fonte_campo,
    cor_texto,
    cor_linha,
    altura_linha: int,
    registro: dict | None = None,
) -> int:
    largura_caixa = (largura_util - 20) // 2
    altura_caixa = altura_linha + 18
    itens = list(campos.items())

    for indice, (chave, valor) in enumerate(itens):
        coluna = indice % 2
        linha = indice // 2
        cx = x + coluna * (largura_caixa + 20)
        cy = y + linha * (altura_caixa + 12)

        desenho.rectangle([cx, cy, cx + largura_caixa, cy + altura_caixa], outline=cor_linha, width=1)
        desenho.text((cx + 10, cy + 8), f"{chave}:", fill=cor_texto, font=fonte_campo)
        if valor:
            valor_txt = str(valor)
            largura_valor = _largura_texto(desenho, valor_txt, fonte_campo)
            while largura_valor > largura_caixa - 24 and len(valor_txt) > 4:
                valor_txt = valor_txt[:-2]
                largura_valor = _largura_texto(desenho, valor_txt, fonte_campo)
            desenho.text((cx + 10, cy + 8 + altura_linha - 4), valor_txt, fill=cor_texto, font=fonte_campo)
            if registro is not None: registro[chave] = valor_txt

    linhas = (len(itens) + 1) // 2
    return y + linhas * (altura_caixa + 12)

def _render_duas_colunas(
    desenho: ImageDraw.ImageDraw,
    campos: dict,
    x: int,
    y: int,
    largura_util: int,
    fonte_campo,
    cor_texto,
    cor_linha,
    altura_linha: int,
    registro: dict | None = None,
) -> int:
    itens = list(campos.items())
    metade = (len(itens) + 1) // 2
    colunas = [itens[:metade], itens[metade:]]
    largura_coluna = (largura_util - 30) // 2

    for indice_coluna, coluna in enumerate(colunas):
        cx = x + indice_coluna * (largura_coluna + 30)
        cy = y
        for chave, valor in coluna:
            rotulo = f"{chave}:"
            desenho.text((cx, cy), rotulo, fill=cor_texto, font=fonte_campo)
            largura_rotulo = _largura_texto(desenho, rotulo, fonte_campo)
            if valor:
                valor_txt = str(valor)
                largura_valor = _largura_texto(desenho, valor_txt, fonte_campo)
                while largura_valor > largura_coluna - largura_rotulo - 12 and len(valor_txt) > 4:
                    valor_txt = valor_txt[:-2]
                    largura_valor = _largura_texto(desenho, valor_txt, fonte_campo)
                desenho.text((cx + largura_rotulo + 8, cy), valor_txt, fill=cor_texto, font=fonte_campo)
                if registro is not None: registro[chave] = valor_txt
            cy += altura_linha
        # linha vertical separando colunas
        if indice_coluna == 0 and len(colunas) > 1:
            sep_x = x + largura_coluna + 15
            desenho.line([sep_x, y - 5, sep_x, y + max(len(c) for c in colunas) * altura_linha],
                         fill=cor_linha, width=1)

    max_linhas = max(len(c) for c in colunas) if colunas else 0
    return y + max_linhas * altura_linha

def _render_tabela(
    desenho: ImageDraw.ImageDraw,
    campos: dict,
    x: int,
    y: int,
    largura_util: int,
    fonte_campo,
    cor_texto,
    cor_linha,
    altura_linha: int,
    registro: dict | None = None,
) -> int:
    itens = list(campos.items())
    largura_rotulo = max(
        (int(_largura_texto(desenho, f"{k}:", fonte_campo)) for k, _ in itens),
        default=80,
    ) + 16
    largura_rotulo = min(largura_rotulo, largura_util // 2)
    largura_valor = largura_util - largura_rotulo

    y_inicio = y
    for indice, (chave, valor) in enumerate(itens):
        if indice % 2 == 1:
            desenho.rectangle([x, y - 4, x + largura_util, y + altura_linha - 6], fill=(238, 238, 238))
        desenho.text((x + 6, y), f"{chave}:", fill=cor_texto, font=fonte_campo)
        if valor:
            valor_txt = str(valor)
            largura_valor_txt = _largura_texto(desenho, valor_txt, fonte_campo)
            while largura_valor_txt > largura_valor - 16 and len(valor_txt) > 4:
                valor_txt = valor_txt[:-2]
                largura_valor_txt = _largura_texto(desenho, valor_txt, fonte_campo)
            desenho.text((x + largura_rotulo, y), valor_txt, fill=cor_texto, font=fonte_campo)
            if registro is not None:
                registro[chave] = valor_txt
        y += altura_linha
        desenho.line([x, y - 4, x + largura_util, y - 4], fill=cor_linha, width=1)

    # bordas laterais
    desenho.line([x + largura_rotulo - 6, y_inicio - 4, x + largura_rotulo - 6, y - 4],
                 fill=cor_linha, width=1)
    return y

def desenhar_cabecalho(
    rng: random.Random,
    desenho: ImageDraw.ImageDraw,
    titulo: str,
    subtitulo: str,
    x: int,
    y: int,
    largura_util: int,
    fonte_titulo,
    fonte_sub,
    cor_titulo,
    cor_texto,
    cor_linha,
) -> None:
    alinhamento = rng.choice(["esquerda", "esquerda", "centro"])

    largura_titulo = _largura_texto(desenho, titulo, fonte_titulo)
    if alinhamento == "centro":
        tx = x + max(0, (largura_util - largura_titulo) // 2)
    else:
        tx = x

    desenho.text((tx, y), titulo, fill=cor_titulo, font=fonte_titulo)
    y += int(fonte_titulo.size * 1.5)

    # Subtítulo
    largura_sub = _largura_texto(desenho, subtitulo, fonte_sub)
    sx = tx
    if alinhamento == "centro": sx = x + max(0, (largura_util - largura_sub) // 2)
    desenho.text((sx, y), subtitulo, fill=cor_texto, font=fonte_sub)
    y += int(fonte_sub.size * 1.7)

    # Selo/código no canto superior direito
    if rng.random() < 0.5:
        fonte_selo = carregar_fonte(rng.choice([13, 15, 16]), "arialbd.ttf")
        selo = rng.choice([
            "CONFIDENCIAL", "INTERNO", "PÚBLICO", "URGENTE",
            "RASCUNHO", "CONFERIDO",
        ])
        largura_selo = _largura_texto(desenho, selo, fonte_selo)
        desenho.text(
            (x + largura_util - largura_selo, y - int(fonte_titulo.size * 1.5)),
            selo, fill=cor_titulo, font=fonte_selo,
        )

    # Linha dupla decorativa
    if rng.random() < 0.4:
        desenho.line([x, y, x + largura_util, y], fill=cor_linha, width=1)
        desenho.line([x, y + 3, x + largura_util, y + 3], fill=cor_linha, width=1)

def desenhar_codigo_barras(
    rng: random.Random,
    desenho: ImageDraw.ImageDraw,
    codigo: str,
    x: int,
    y: int,
    largura_util: int,
    cor_texto,
    estilo: dict,
) -> None:
    altura_barras = rng.randint(24, 40)
    largura_total = rng.randint(120, 260)
    cursor = x

    while cursor < x + largura_total:
        largura_barra = rng.choice([1, 1, 2, 3])
        if rng.random() < 0.55:
            desenho.rectangle([cursor, y, cursor + largura_barra, y + altura_barras], fill=cor_texto)
        cursor += largura_barra + rng.choice([1, 2, 3])

    fonte_codigo = carregar_fonte(11, estilo["fonte"])
    desenho.text((x, y + altura_barras + 2), codigo, fill=cor_texto, font=fonte_codigo)

def recortar_para_conteudo(imagem: Image.Image, cor_fundo: tuple, margem_extra: int = 24) -> Image.Image:
    largura, altura = imagem.size

    if np is None: return imagem

    # Varre de baixo para cima procurando a última linha "com conteúdo"
    # (pixels diferentes do fundo), usando uma tolerância para ruído leve.
    fundo = np.asarray(cor_fundo).astype("int16")
    array = np.asarray(imagem).astype("int16")

    diferenca = np.abs(array - fundo).sum(axis=2)
    linhas_com_conteudo = np.where(diferenca.max(axis=1) > 30)[0]

    if linhas_com_conteudo.size == 0: return imagem

    ultima_linha = int(linhas_com_conteudo.max())
    nova_altura = min(altura, ultima_linha + 1 + margem_extra)

    # Não recorta mais que 70% da altura (documentos muito cheios ficam intactos)
    nova_altura = max(nova_altura, int(altura * 0.3))

    if nova_altura >= altura: return imagem

    return imagem.crop((0, 0, largura, nova_altura))

def aplicar_degradacoes(rng: random.Random, imagem: Image.Image, estilo: dict) -> Image.Image:
    # Ruído
    if np is not None and rng.random() < 0.35:
        array = np.asarray(imagem).astype(np.int16)
        sigma = rng.uniform(3, 12)
        ruido = np.random.normal(0, sigma, array.shape)
        array = np.clip(array + ruido, 0, 255).astype("uint8")
        imagem = Image.fromarray(array)

    # Desfoque leve (simula leitura de scanner)
    if rng.random() < 0.2: imagem = imagem.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))

    # Rotação leve
    if rng.random() < 0.3:
        angulo = rng.uniform(-2.5, 2.5)
        imagem = imagem.rotate(angulo, resample=Image.BICUBIC, expand=False, fillcolor=estilo["cor_fundo"])

    # Escurecimento (bordas / sombra)
    if np is not None and rng.random() < 0.2:
        array = np.asarray(imagem).astype("float32")
        largura, altura = imagem.size
        yy, xx = np.mgrid[0:altura, 0:largura]
        centro_x, centro_y = largura / 2, altura / 2
        dist = np.sqrt(((xx - centro_x) / centro_x) ** 2 + ((yy - centro_y) / centro_y) ** 2)
        fator = 1.0 - rng.uniform(0.05, 0.22) * np.clip(dist, 0, 1.5)
        array = np.clip(array * fator[:, :, None], 0, 255).astype("uint8")
        imagem = Image.fromarray(array)

    # Contraste/ brilho
    if rng.random() < 0.25:
        from PIL import ImageEnhance
        imagem = ImageEnhance.Contrast(imagem).enhance(rng.uniform(0.8, 1.25))
        imagem = ImageEnhance.Brightness(imagem).enhance(rng.uniform(0.9, 1.1))

    return imagem

def gerar_documento(
    rng: random.Random,
    indice: int,
    dir_saida: Path,
    tipo_doc: str | None = None,
    forcar_layout: str | None = None,
) -> dict:
    if tipo_doc is None: tipo_doc = rng.choice(list(TIPOS_DOCUMENTO.keys()))
    perfil = TIPOS_DOCUMENTO[tipo_doc]

    estilo = dict(rng.choice(ESTILOS_VISUAIS))
    layout = forcar_layout or rng.choice(LAYOUTS)

    campos = gerar_campos_aleatorios(rng, tipo_doc)
    titulo = rng.choice(perfil["titulos"])
    subtitulo = rng.choice(perfil["subtitulos"])

    # Dimensões da imagem
    largura = rng.choice([720, 800, 900, 1000, 1100])
    margem = rng.randint(28, 60)
    altura = rng.randint(560, 900)
    altura = max(altura, margem * 2 + 260 + len(campos) * 34)

    fonte_titulo = carregar_fonte(rng.choice([24, 28, 32, 36]), estilo["fonte"])
    fonte_sub = carregar_fonte(rng.choice([15, 17, 19]), estilo["fonte"])
    fonte_campo = carregar_fonte(rng.choice([15, 17, 18, 20]), estilo["fonte"])

    fundo = estilo["cor_fundo"]
    if estilo["fundo_cinza"]: fundo = tuple(max(0, c - rng.randint(0, 12)) for c in fundo)

    imagem = Image.new("RGB", (largura, altura), fundo)
    desenho = ImageDraw.Draw(imagem)

    cor_texto = estilo["cor_texto"]
    cor_linha = tuple(min(255, c + 60) for c in cor_texto)
    cor_titulo = rng.choice(CORES_TITULO)
    altura_linha = max(24, int(fonte_campo.size * 1.7))

    x = margem
    largura_util = largura - margem * 2
    y = margem

    # Cabeçalho
    desenhar_cabecalho(rng, desenho, titulo, subtitulo, x, y, largura_util, fonte_titulo, fonte_sub, cor_titulo, cor_texto, cor_linha)
    y += rng.choice([70, 84, 96])

    # Linha divisória do cabeçalho
    if rng.random() < 0.7: desenho.line([x, y - 16, x + largura_util, y - 16], fill=cor_linha, width=2)

    # Campos conforme o layout
    renderizadores = {
        "lista": _render_lista,
        "tabela": _render_tabela,
        "duas_colunas": _render_duas_colunas,
        "blocos": _render_blocos,
    }

    # `registro` guarda exatamente o texto desenhado na imagem (pode ter sido
    # truncado para caber na largura). A anotação usa esses mesmos valores,
    # garantindo que imagem e rótulo nunca fiquem inconsistentes.
    registro: dict[str, str] = {}

    if layout == "lista_alternada":
        y_fim = _render_lista_alternada(
            desenho, campos, x, y, largura_util, fonte_campo,
            cor_texto, cor_linha, altura_linha,
            tuple(max(0, c - 12) for c in fundo),
            registro,
        )
    elif layout in renderizadores:
        y_fim = renderizadores[layout](
            desenho, campos, x, y, largura_util, fonte_campo,
            cor_texto, cor_linha, altura_linha, registro,
        )
    else:
        y_fim = y

    # Garante que todo campo sorteado esteja na anotação, mesmo se o layout
    # não renderizou (ex.: valor vazio)
    for chave, valor in campos.items(): registro.setdefault(chave, str(valor))

    # Rodapé opcional
    if rng.random() < 0.5:
        y_rodape = min(altura - margem - 20, y_fim + 20)
        fonte_rodape = carregar_fonte(12, estilo["fonte"])
        rodape = rng.choice([
            "Documento processado eletronicamente.",
            "Página 1 de 1",
            "Emitido em sistema interno.",
            "Válido apenas com assinatura.",
        ])
        desenho.text((x, y_rodape), rodape, fill=cor_texto, font=fonte_rodape)
        y_rodape += 18
        if rng.random() < 0.6:
            codigo = f"REF-{rng.randint(10000, 99999)}"
            desenhar_codigo_barras(rng, desenho, codigo, x, y_rodape, largura_util, cor_texto, estilo)

    # Recorta o espaço em branco inferior, mantendo uma margem final
    imagem = recortar_para_conteudo(imagem, fundo, margem_extra=rng.randint(16, 40))

    # Degradações visuais
    imagem = aplicar_degradacoes(rng, imagem, estilo)

    nome_imagem = f"doc_{indice:06d}.png"
    caminho_imagem = dir_saida / "images" / nome_imagem
    caminho_imagem.parent.mkdir(parents=True, exist_ok=True)
    imagem.save(caminho_imagem, "PNG", quality=95)

    anotacao = {"image": nome_imagem, "fields": registro}

    dir_anotacoes = dir_saida / "annotations"
    dir_anotacoes.mkdir(parents=True, exist_ok=True)
    caminho_anotacao = dir_anotacoes / f"{caminho_imagem.stem}.json"
    with open(caminho_anotacao, "w", encoding="utf-8") as f:
        json.dump(anotacao, f, ensure_ascii=False, indent=2)

    return anotacao

def main():
    parser = argparse.ArgumentParser(description="Gera documentos sintéticos genéricos (imagem + JSON) para treinar o Donut.")
    parser.add_argument("--output", "-o", default="./training_data", help="Diretório de saída (padrão: ./training_data)")
    parser.add_argument("--num", "-n", type=int, default=1000000, help="Número de documentos a gerar (padrão: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Semente do gerador aleatório (padrão: 42)")
    parser.add_argument("--types", default="all", help=("Tipos de documento a gerar, separados por vírgula (ex: financeiro,logistica). Use 'all' para todos (padrão: all)"))
    parser.add_argument("--layout", default="all", help=("Layout fixo (lista, lista_alternada, duas_colunas, tabela, blocos) ou 'all' para variar (padrão: all)"))
    parser.add_argument("--balanced-layouts", action="store_true", help="Distribui documentos igualmente entre todos os layouts (quando --layout all)",)
    parser.add_argument("--balanced-types", action="store_true", help="Distribui documentos igualmente entre todos os tipos (quando --types all)")
    parser.add_argument("--start-index", type=int, default=0, help="Índice inicial da numeração dos arquivos (padrão: 0)")
    parser.add_argument("--limpar", action="store_true", help="Apaga imagens/anotações anteriores na pasta de saída antes de gerar",)

    args = parser.parse_args()

    if args.types == "all":
        tipos = list(TIPOS_DOCUMENTO.keys())
    else:
        tipos = [t.strip() for t in args.types.split(",") if t.strip()]
        invalidos = [t for t in tipos if t not in TIPOS_DOCUMENTO]
        if invalidos:
            print(f"❌ Tipos inválidos: {', '.join(invalidos)}")
            print(f"   Tipos disponíveis: {', '.join(TIPOS_DOCUMENTO.keys())}")
            sys.exit(1)

    if args.layout != "all" and args.layout not in LAYOUTS:
        print(f"❌ Layout inválido: {args.layout}")
        print(f"   Layouts disponíveis: {', '.join(LAYOUTS)}")
        sys.exit(1)

    dir_saida = Path(args.output)
    (dir_saida / "images").mkdir(parents=True, exist_ok=True)
    (dir_saida / "annotations").mkdir(parents=True, exist_ok=True)

    if args.limpar:
        for pasta in ("images", "annotations"):
            destino = dir_saida / pasta
            for arquivo in destino.glob("*"):
                if arquivo.is_file(): arquivo.unlink()
        print(f"🧹 Conteúdo anterior removido de {dir_saida}")

    # Verifica quantas imagens de treino já existem na pasta de saída
    dir_imagens = dir_saida / "images"
    imagens_existentes = list(dir_imagens.glob("doc_*.png"))
    qtd_existente = len(imagens_existentes)

    # Determina o próximo índice disponível (evita sobrescrever arquivos existentes)
    proximo_indice = args.start_index
    if imagens_existentes:
        indices_existentes = []
        for arq in imagens_existentes:
            try:
                # nome do arquivo: doc_000001.png
                indice_arq = int(arq.stem.split("_")[1])
                indices_existentes.append(indice_arq)
            except (ValueError, IndexError):
                continue
        if indices_existentes:
            proximo_indice = max(proximo_indice, max(indices_existentes) + 1)

    # Calcula quantos documentos ainda precisam ser gerados
    restante = args.num - qtd_existente
    if restante <= 0:
        print("=" * 60)
        print("📄 Gerador de documentos sintéticos genéricos")
        print("=" * 60)
        print(f"Saída: {dir_saida}")
        print(f"Quantidade solicitada: {args.num}")
        print(f"Imagens já existentes: {qtd_existente}")
        print("✅ Nenhum documento novo necessário. O alvo já foi atingido.")
        print("=" * 60)
        return

    rng = random.Random(args.seed)

    print("=" * 60)
    print("📄 Gerador de documentos sintéticos genéricos")
    print("=" * 60)
    print(f"Saída: {dir_saida}")
    print(f"Quantidade solicitada: {args.num}")
    print(f"Imagens já existentes: {qtd_existente}")
    print(f"Documentos a gerar: {restante}")
    print(f"Índice inicial: {proximo_indice}")
    print(f"Tipos: {', '.join(tipos)}")
    print(f"Layouts: {args.layout}")
    if args.balanced_layouts: print(f"Layouts balanceados: Sim (round-robin)")
    if args.balanced_types: print(f"Tipos balanceados: Sim (round-robin)")
    print(f"Semente: {args.seed}")
    print("=" * 60)

    campos_vistos: set[str] = set()
    contagem_tipos: dict[str, int] = {}
    contagem_layouts: dict[str, int] = {}

    # Para distribuição balanceada, cicla pelos tipos e layouts em round-robin.
    # Isso garante que o modelo veja todas as combinações tipo+layout,
    # evitando viés para um layout específico e forçando aprendizado geral.
    tipo_cycle = tipos if args.balanced_types else None
    layout_cycle = LAYOUTS if (args.balanced_layouts and args.layout == "all") else None

    for i in tqdm(range(restante), desc="Gerando documentos", unit="doc"):
        idx_global = i  # posição relativa dentro deste lote
        if tipo_cycle is not None:
            tipo = tipo_cycle[idx_global % len(tipo_cycle)]
        else:
            tipo = rng.choice(tipos)

        if layout_cycle is not None:
            layout = layout_cycle[idx_global % len(layout_cycle)]
        elif args.layout != "all":
            layout = args.layout
        else:
            layout = None  # aleatório dentro de gerar_documento

        anotacao = gerar_documento(
            rng,
            indice=proximo_indice + i,
            dir_saida=dir_saida,
            tipo_doc=tipo,
            forcar_layout=layout,
        )
        campos_vistos.update(anotacao["fields"].keys())
        contagem_tipos[tipo] = contagem_tipos.get(tipo, 0) + 1
        if layout is not None:
            contagem_layouts[layout] = contagem_layouts.get(layout, 0) + 1
        else:
            contagem_layouts["<aleatorio>"] = contagem_layouts.get("<aleatorio>", 0) + 1

    print()
    print("=" * 60)
    print("✅ Geração concluída!")
    print(f"Imagens: {dir_saida / 'images'}")
    print(f"Anotações: {dir_saida / 'annotations'}")
    print(f"Documentos: {args.num}")
    print(f"Rótulos distintos: {len(campos_vistos)}")
    print("Distribuição por tipo:")
    for tipo, qtd in sorted(contagem_tipos.items(), key=lambda kv: -kv[1]):
        print(f"     - {tipo}: {qtd}")
    print("Distribuição por layout:")
    for layout, qtd in sorted(contagem_layouts.items(), key=lambda kv: -kv[1]):
        print(f"     - {layout}: {qtd}")
    print("=" * 60)
    print()
    print("Próximos passos:")
    print(f"1. Verifique o dataset: py training/prepare_dataset.py --verify {dir_saida / 'images'} {dir_saida / 'annotations'}")
    print(f"2. Treine o modelo:    py main.py train --images {dir_saida / 'images'} --annotations {dir_saida / 'annotations'}")

if __name__ == "__main__":
    main()
