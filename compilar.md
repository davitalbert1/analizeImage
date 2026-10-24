# Como compilar (C++)

Este documento cobre a compilação do gerador sintético em C++ ([gerar_itens_treino.cpp](gerar_itens_treino.cpp)) e dos exemplos da pasta `examples/`.

> A pasta existente no repositório é `examples` (com "a"), não `exemples`. O CMake já compila os dois arquivos dela.

## Alvos gerados

| Alvo | Fonte | Dependências |
|------|-------|--------------|
| `gerar_itens_treino` | `gerar_itens_treino.cpp` | OpenCV (core, imgproc, imgcodecs, freetype), nlohmann/json |
| `create_test_doc` | `examples/create_test_doc.cpp` | OpenCV |
| `example_usage` | `examples/example_usage.cpp` | Tesseract, Leptonica, nlohmann/json |

`example_usage` só é compilado se `pkg-config`, Tesseract e Leptonica forem encontrados; caso contrário o CMake emite um aviso e continua com os outros alvos.

## 1. Ambiente (Windows + MSYS2)

Use o shell **MSYS2 MINGW64** (não o Git Bash nem o `cmd`). Ele pode ser aberto por `D:\msys64\mingw64.exe` ou pelo atalho do MSYS2 "MINGW64".

Instale o toolchain e as bibliotecas:

```bash
pacman -S --needed \
  mingw-w64-x86_64-gcc \
  mingw-w64-x86_64-cmake \
  mingw-w64-x86_64-ninja \
  mingw-w64-x86_64-opencv \
  mingw-w64-x86_64-nlohmann-json \
  mingw-w64-x86_64-pkgconf \
  mingw-w64-x86_64-tesseract-ocr \
  mingw-w64-x86_64-leptonica
```

Para executar o `example_usage` com OCR em português e inglês, instale também os dados de idioma:

```bash
pacman -S --needed \
  mingw-w64-x86_64-tesseract-data-por \
  mingw-w64-x86_64-tesseract-data-eng
```

O pacote `mingw-w64-x86_64-opencv` do MSYS2 já inclui o módulo `opencv_freetype` (OpenCV contrib), necessário para desenhar texto TTF/UTF-8. Se você usa um OpenCV próprio, ele precisa ter sido compilado com o módulo `freetype`.

## 2. Configurar e compilar

Na raiz do projeto (`/f/analizeImage` no shell MSYS2):

```bash
cd /f/analizeImage

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Se preferir o Make do MinGW:

```bash
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Não troque o gerador dentro de uma pasta `build/` já configurada. Se precisar trocar, use outra pasta (por exemplo, `build-make`) ou apague `build/` antes de reconfigurar.

### Opções do CMake

| Opção | Padrão | Efeito |
|-------|--------|--------|
| `BUILD_EXAMPLES` | `ON` | Compila a pasta `examples/` |
| `CMAKE_BUILD_TYPE` | `Release` | Tipo de build padrão |
| `OpenCV_DIR` | detectado | Caminho do `OpenCVConfig.cmake` de um OpenCV customizado |

Exemplos:

```bash
# Compilar tudo
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Compilar apenas o gerador
cmake -S . -B build -G Ninja -DBUILD_EXAMPLES=OFF
cmake --build build

# Compilar apenas os exemplos (depois da configuração)
cmake --build build --target exemplos
```

Com OpenCV fora do MSYS2:

```bash
cmake -S . -B build -G Ninja \
  -DOpenCV_DIR="C:/opencv/build/lib/cmake/opencv4"
```

## 3. Executar

Os binários ficam em `build/` e `build/examples/`:

```bash
./build/gerar_itens_treino.exe --help
./build/gerar_itens_treino.exe -o training_data -n 100 --seed 42

./build/examples/create_test_doc.exe
./build/examples/example_usage.exe
```

Executando pelo próprio shell MINGW64, as DLLs de `/mingw64/bin` já estão no `PATH`. Para rodar pelo `cmd` ou PowerShell, adicione a pasta ao `PATH`:

```cmd
set PATH=D:\msys64\mingw64\bin;%PATH%
```

```powershell
$env:PATH = "D:\msys64\mingw64\bin;$env:PATH"
```

Se o `example_usage` reclamar de idiomas do Tesseract, aponte o `TESSDATA_PREFIX` para a pasta de dados:

```bash
export TESSDATA_PREFIX=/mingw64/share/tessdata
```

No `cmd`:

```cmd
set TESSDATA_PREFIX=D:\msys64\mingw64\share\tessdata
```

## 4. Usar o gerador

O comportamento e os parâmetros são equivalentes aos do script Python.

| Opção | Padrão | Descrição |
|-------|--------|-----------|
| `-o`, `--output DIR` | `./training_data` | Pasta de saída |
| `-n`, `--num N` | `1000000` | Número total de documentos desejado |
| `--seed N` | `42` | Semente do gerador |
| `--types LISTA` | `all` | Tipos separados por vírgula |
| `--layout NOME` | `all` | `lista`, `lista_alternada`, `duas_colunas`, `tabela`, `blocos` ou `all` |
| `--balanced-layouts` | desligado | Round-robin de layouts quando `--layout all` |
| `--balanced-types` | desligado | Round-robin de tipos |
| `--start-index N` | `0` | Índice inicial dos arquivos |
| `--limpar` | desligado | Apaga imagens/anotações anteriores da saída |

Tipos disponíveis: `financeiro`, `transporte`, `recursos_humanos`, `saude`, `logistica`, `energia`, `telecom`, `seguranca`, `academico`, `generico`.

### Exemplos

```bash
# 500 documentos variados, tipos e layouts balanceados
./build/gerar_itens_treino.exe -o training_data -n 500 --balanced-types --balanced-layouts

# Somente tabelas de dois tipos
./build/gerar_itens_treino.exe -o training_data -n 200 \
  --types financeiro,logistica --layout tabela

# Recomeçar do zero com 1.000 documentos
./build/gerar_itens_treino.exe -o training_data -n 1000 --limpar

# Continuar um dataset existente (conta os PNGs e segue do maior índice + 1)
./build/gerar_itens_treino.exe -o training_data -n 2000
```

O default de `--num` é **1.000.000** (o mesmo default efetivo do script Python). Para testes, sempre informe `-n` com um valor pequeno.

### Saída

```text
training_data/
├── images/
│   ├── doc_000000.png
│   ├── doc_000001.png
│   └── ...
└── annotations/
    ├── doc_000000.json
    ├── doc_000001.json
    └── ...
```

Cada JSON segue o formato usado pelo treinamento:

```json
{
  "image": "doc_000000.png",
  "fields": {
    "Código": "A1234-56",
    "Data": "12/03/2026"
  }
}
```

Durante a geração o console mostra quantidade, percentual, documentos por segundo, tempo decorrido e ETA. As degradações (ruído, desfoque, rotação, vinheta) são processadas em CPU e dominam o custo por documento.

## 5. Fontes TTF

O desenho de texto usa `cv::freetype`, com busca automática nesta ordem:

1. `./fonts/`
2. `%WINDIR%\Fonts` (normalmente `C:\Windows\Fonts`)
3. `%SystemRoot%\Fonts`
4. `%LOCALAPPDATA%\Microsoft\Windows\Fonts`
5. Diretórios usuais de Linux/macOS (`/usr/share/fonts`, `~/.fonts`, etc.)

As fontes preferidas são `arial.ttf`, `arialbd.ttf`, `calibri.ttf`, `times.ttf`, `verdana.ttf`, `consola.ttf` e `cour.ttf`. Se alguma não existir, o programa tenta `segoeui.ttf`, `DejaVuSans.ttf` e `DejaVuSansMono.ttf`.

Se nenhuma fonte for encontrada, copie uma TTF para `fonts/`:

```bash
mkdir -p fonts
cp /c/Windows/Fonts/arial.ttf fonts/
```

## 6. Problemas comuns

**`cmake` não encontrado**
Abra o shell MINGW64 do MSYS2. O `cmake` do MSYS2 fica em `D:\msys64\mingw64\bin`.

**`Could NOT find OpenCV` ou componente `freetype` ausente**
Instale `mingw-w64-x86_64-opencv` (o pacote oficial inclui o módulo freetype) e reconfigure com uma pasta de build limpa:

```bash
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

**`nlohmann/json.hpp` não encontrado**
Instale `mingw-w64-x86_64-nlohmann-json` ou copie o header para um diretório incluído pelo compilador.

**`example_usage` não foi compilado**
O CMake avisou que `pkg-config`, Tesseract ou Leptonica não foram encontrados. Instale os três pacotes da seção 1 e reconfigure.

**Erro de DLL ao executar (`libopencv_core-*.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll`)**
Rode pelo shell MINGW64 ou adicione `D:\msys64\mingw64\bin` ao `PATH`.

**Acentos quebrados no console**
O programa configura o console para UTF-8 no Windows. Use o Windows Terminal ou um console com fonte que suporte acentos. Alternativamente, execute `chcp 65001` antes.

**Gerador muito lento**
Reduza `-n` para validar o fluxo. Cada documento passa por desenho, recorte e degradações; a geração de um milhão de imagens é uma tarefa longa por natureza.

## 7. Próximos passos (treinamento)

O gerador C++ produz o mesmo dataset que o script Python. Depois de gerar:

```bash
py training/prepare_dataset.py --verify training_data/images training_data/annotations
py main.py train --images training_data/images --annotations training_data/annotations
```

O script Python original (`gerar_itens_treino.py`) continua disponível e não foi removido.
