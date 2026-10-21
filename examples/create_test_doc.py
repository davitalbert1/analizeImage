from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_test_document():
    width, height = 800, 1100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Tentar usar fonte com tamanho adequado
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_header = ImageFont.truetype("arial.ttf", 18)
        font_normal = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font_title = ImageFont.load_default()
        font_header = font_title
        font_normal = font_title
        font_small = font_title

    y = 30

    # Cabeçalho
    draw.text((250, y), "LABORATORIO SAUDE", fill="navy", font=font_title)
    y += 35
    draw.text((280, y), "Resultado de Exames", fill="gray", font=font_header)
    y += 40

    # Linha separadora
    draw.line([(50, y), (750, y)], fill="navy", width=2)
    y += 20

    # Dados do paciente
    draw.text((60, y), "Paciente:", fill="black", font=font_normal)
    draw.text((200, y), "Maria Santos Silva", fill="black", font=font_normal)
    y += 25
    draw.text((60, y), "Data:", fill="black", font=font_normal)
    draw.text((200, y), "15/10/2026", fill="black", font=font_normal)
    y += 25
    draw.text((60, y), "Idade:", fill="black", font=font_normal)
    draw.text((200, y), "45 anos", fill="black", font=font_normal)
    y += 25
    draw.text((60, y), "Medico:", fill="black", font=font_normal)
    draw.text((200, y), "Dr. Carlos Mendes - CRM 12345", fill="black", font=font_normal)
    y += 35

    # Linha separadora
    draw.line([(50, y), (750, y)], fill="gray", width=1)
    y += 15

    # Cabeçalho da tabela
    draw.text((60, y), "BIOQUIMICA", fill="navy", font=font_header)
    y += 30

    # Cabeçalho das colunas
    draw.text((60, y), "Exame", fill="gray", font=font_small)
    draw.text((300, y), "Resultado", fill="gray", font=font_small)
    draw.text((450, y), "Unidade", fill="gray", font=font_small)
    draw.text((580, y), "Ref.", fill="gray", font=font_small)
    y += 5
    draw.line([(50, y + 15), (750, y + 15)], fill="lightgray", width=1)
    y += 25

    # Dados dos exames
    exams = [
        ("Glicose", "95", "mg/dL", "70 a 99"),
        ("Colesterol Total", "180", "mg/dL", "< 200"),
        ("HDL Colesterol", "55", "mg/dL", "> 40"),
        ("LDL Colesterol", "105", "mg/dL", "< 130"),
        ("Triglicerideos", "120", "mg/dL", "< 150"),
        ("Ureia", "32", "mg/dL", "15 a 45"),
        ("Creatinina", "0.9", "mg/dL", "0.6 a 1.2"),
        ("Acido Urico", "5.2", "mg/dL", "2.5 a 7.0"),
        ("TGO (AST)", "25", "U/L", "< 40"),
        ("TGP (ALT)", "30", "U/L", "< 41"),
    ]

    for exam_name, result, unit, ref in exams:
        draw.text((60, y), exam_name, fill="black", font=font_normal)
        draw.text((300, y), result, fill="black", font=font_normal)
        draw.text((450, y), unit, fill="black", font=font_normal)
        draw.text((580, y), ref, fill="gray", font=font_small)
        y += 25
        draw.line([(50, y + 2), (750, y + 2)], fill="#eeeeee", width=1)
        y += 8

    y += 15
    draw.line([(50, y), (750, y)], fill="gray", width=1)
    y += 15

    # Hemograma
    draw.text((60, y), "HEMOGRAMA", fill="navy", font=font_header)
    y += 30

    draw.text((60, y), "Exame", fill="gray", font=font_small)
    draw.text((300, y), "Resultado", fill="gray", font=font_small)
    draw.text((450, y), "Unidade", fill="gray", font=font_small)
    draw.text((580, y), "Ref.", fill="gray", font=font_small)
    y += 5
    draw.line([(50, y + 15), (750, y + 15)], fill="lightgray", width=1)
    y += 25

    hemograma = [
        ("Hemoglobina", "14.2", "g/dL", "12.0 a 16.0"),
        ("Hematocrito", "42.5", "%", "36 a 46"),
        ("Leucocitos", "7.500", "/mm3", "4.000 a 11.000"),
        ("Plaquetas", "250.000", "/mm3", "150.000 a 400.000"),
        ("Hemacia", "4.8", "milhoes/mm3", "4.0 a 5.5"),
        ("VCM", "88", "fL", "80 a 100"),
    ]

    for exam_name, result, unit, ref in hemograma:
        draw.text((60, y), exam_name, fill="black", font=font_normal)
        draw.text((300, y), result, fill="black", font=font_normal)
        draw.text((450, y), unit, fill="black", font=font_normal)
        draw.text((580, y), ref, fill="gray", font=font_small)
        y += 25
        draw.line([(50, y + 2), (750, y + 2)], fill="#eeeeee", width=1)
        y += 8

    y += 20
    draw.line([(50, y), (750, y)], fill="navy", width=2)
    y += 15
    draw.text((200, y), "Responsavel Tecnico: Dra. Ana Oliveira", fill="gray", font=font_small)

    # Salvar
    output_path = Path(__file__).parent.parent / "test_docs" / "exame_sangue_teste.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Documento de teste criado: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_document()
