
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ============================================================
// Utilitarios
// ============================================================

std::string toLower(std::string text) {
    std::transform(
        text.begin(), text.end(), text.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        }
    );

    return text;
}

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");

    if (first == std::string::npos) {
        return "";
    }

    const auto last = text.find_last_not_of(" \t\r\n");

    return text.substr(first, last - first + 1);
}

// ============================================================
// Resultado da extracao
// ============================================================

struct ExtractionResult {
    std::string filePath;
    std::map<std::string, std::string> fields;
    std::string rawText;

    void printSummary() const {
        std::cout << "\n========================================\n";
        std::cout << "Arquivo: " << filePath << '\n';
        std::cout << "========================================\n";

        if (fields.empty()) {
            std::cout
                << "Nenhum campo identificado automaticamente.\n";
        }

        for (const auto& [field, value] : fields) {
            std::cout << field << ": " << value << '\n';
        }

        std::cout << "\nTexto extraido pelo OCR:\n";
        std::cout << rawText << '\n';
    }

    std::string getField(const std::string& name) const {
        const std::string target = toLower(name);

        for (const auto& [key, value] : fields) {
            if (toLower(key) == target) {
                return value;
            }
        }

        return "";
    }

    std::map<std::string, std::string>
    searchFields(const std::string& keyword) const {
        std::map<std::string, std::string> matches;
        const std::string target = toLower(keyword);

        for (const auto& [key, value] : fields) {
            if (toLower(key).find(target) != std::string::npos ||
                toLower(value).find(target) != std::string::npos) {
                matches[key] = value;
            }
        }

        return matches;
    }

    std::string getRawText() const {
        return rawText;
    }

    std::string toJson() const {
        json output;

        output["file_path"] = filePath;
        output["fields"] = fields;
        output["raw_text"] = rawText;

        return output.dump(4);
    }
};

// ============================================================
// Deleter para imagens Leptonica
// ============================================================

struct PixDeleter {
    void operator()(PIX* image) const {
        if (image) {
            pixDestroy(&image);
        }
    }
};

// ============================================================
// Extrator de documentos
// ============================================================

class DocumentExtractor {
public:
    explicit DocumentExtractor(
        const std::string& ocrLanguages = "por+eng"
    ) {
        if (ocr_.Init(nullptr, ocrLanguages.c_str()) != 0) {
            throw std::runtime_error(
                "Falha ao inicializar o Tesseract. "
                "Verifique os idiomas instalados."
            );
        }

        ocr_.SetPageSegMode(tesseract::PSM_AUTO);
    }

    ~DocumentExtractor() {
        ocr_.End();
    }

    DocumentExtractor(const DocumentExtractor&) = delete;
    DocumentExtractor& operator=(const DocumentExtractor&) = delete;

    // --------------------------------------------------------
    // Extrair um documento
    // --------------------------------------------------------

    ExtractionResult extract(const fs::path& filePath) {
        if (!fs::exists(filePath) ||
            !fs::is_regular_file(filePath)) {
            throw std::runtime_error(
                "Arquivo nao encontrado: " + filePath.string()
            );
        }

        const std::string extension =
            toLower(filePath.extension().string());

        if (extension == ".pdf") {
            throw std::runtime_error(
                "PDF nao suportado diretamente nesta versao. "
                "Converta cada pagina para PNG antes de processar."
            );
        }

        std::unique_ptr<PIX, PixDeleter> image(
            pixRead(filePath.string().c_str())
        );

        if (!image) {
            throw std::runtime_error(
                "Nao foi possivel abrir a imagem: " +
                filePath.string()
            );
        }

        ocr_.SetImage(image.get());

        char* recognized = ocr_.GetUTF8Text();

        if (!recognized) {
            ocr_.Clear();

            throw std::runtime_error(
                "O Tesseract nao conseguiu reconhecer o texto."
            );
        }

        ExtractionResult result;
        result.filePath = filePath.string();
        result.rawText = trim(recognized);

        delete[] recognized;

        ocr_.Clear();

        result.fields = identifyFields(result.rawText);

        return result;
    }

    // --------------------------------------------------------
    // Processar varios arquivos
    // --------------------------------------------------------

    std::vector<ExtractionResult> extractBatch(
        const std::vector<fs::path>& files
    ) {
        std::vector<ExtractionResult> results;

        for (const auto& file : files) {
            try {
                results.push_back(extract(file));
            }
            catch (const std::exception& error) {
                std::cerr << "[ERRO] "
                          << file.string() << ": "
                          << error.what() << '\n';
            }
        }

        return results;
    }

    // --------------------------------------------------------
    // Processar diretorio de imagens
    // --------------------------------------------------------

    std::vector<ExtractionResult> extractDirectory(
        const fs::path& directory
    ) {
        if (!fs::exists(directory) ||
            !fs::is_directory(directory)) {
            throw std::runtime_error(
                "Diretorio invalido: " + directory.string()
            );
        }

        std::vector<fs::path> files;

        for (const auto& entry :
             fs::directory_iterator(directory)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const std::string extension =
                toLower(entry.path().extension().string());

            if (extension == ".png" ||
                extension == ".jpg" ||
                extension == ".jpeg" ||
                extension == ".bmp" ||
                extension == ".tif" ||
                extension == ".tiff") {
                files.push_back(entry.path());
            }
        }

        std::sort(files.begin(), files.end());

        return extractBatch(files);
    }

private:
    tesseract::TessBaseAPI ocr_;

    // --------------------------------------------------------
    // Identificar campos conhecidos no texto extraido
    // --------------------------------------------------------

    std::map<std::string, std::string> identifyFields(
        const std::string& text
    ) const {
        std::map<std::string, std::string> fields;

        const std::vector<std::pair<std::string, std::string>>
            patterns = {
            {
                "glicose",
                R"((?:glicose|glicemia)\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?\s*(?:mg/dL|mmol/L)?))"
            },
            {
                "colesterol_total",
                R"((?:colesterol\s+total)\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?\s*(?:mg/dL|mmol/L)?))"
            },
            {
                "hdl",
                R"((?:colesterol\s+hdl|hdl)\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?\s*(?:mg/dL|mmol/L)?))"
            },
            {
                "ldl",
                R"((?:colesterol\s+ldl|ldl)\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?\s*(?:mg/dL|mmol/L)?))"
            },
            {
                "triglicerideos",
                R"((?:triglicer[ií]deos|triglicerides)\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?\s*(?:mg/dL|mmol/L)?))"
            }
        };

        const std::string normalized = toLower(text);

        for (const auto& [field, pattern] : patterns) {
            try {
                const std::regex expression(
                    pattern,
                    std::regex_constants::icase
                );

                std::smatch match;

                if (std::regex_search(
                    normalized, match, expression
                )) {
                    fields[field] = trim(match[1].str());
                }
            }
            catch (const std::regex_error&) {
                // Ignora um padrao invalido.
            }
        }

        return fields;
    }
};

// ============================================================
// Exemplo 1: Extracao basica
// ============================================================

void exampleBasicExtraction(DocumentExtractor& extractor) {
    std::cout << "\n=== Exemplo 1: Extracao basica ===\n";

    const fs::path document = "seu_documento.png";

    if (!fs::exists(document)) {
        std::cout << "Arquivo '" << document
                  << "' nao encontrado.\n";
        return;
    }

    try {
        const ExtractionResult result =
            extractor.extract(document);

        result.printSummary();

        const std::string glucose =
            result.getField("glicose");

        if (!glucose.empty()) {
            std::cout << "\nGlicose: " << glucose << '\n';
        }

        const auto cholesterol =
            result.searchFields("colesterol");

        for (const auto& [field, value] : cholesterol) {
            std::cout << field << ": " << value << '\n';
        }
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
}

// ============================================================
// Exemplo 2: Processamento em lote
// ============================================================

void exampleBatchProcessing(DocumentExtractor& extractor) {
    std::cout << "\n=== Exemplo 2: Processamento em lote ===\n";

    const fs::path directory = "documentos";

    if (!fs::exists(directory)) {
        std::cout << "Diretorio documentos nao encontrado.\n";
        return;
    }

    try {
        const auto results =
            extractor.extractDirectory(directory);

        for (const auto& result : results) {
            std::cout << "\nArquivo: "
                      << fs::path(result.filePath).filename()
                      << '\n';

            for (const auto& [field, value] : result.fields) {
                std::cout << "  " << field
                          << ": " << value << '\n';
            }
        }

        std::cout << "\nTotal processado: "
                  << results.size() << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
}

// ============================================================
// Exemplo 3: Exportar resultado para JSON
// ============================================================

void exampleJsonExport(DocumentExtractor& extractor) {
    std::cout << "\n=== Exemplo 3: Exportacao JSON ===\n";

    const fs::path document = "seu_documento.png";

    if (!fs::exists(document)) {
        std::cout << "Arquivo '" << document
                  << "' nao encontrado.\n";
        return;
    }

    try {
        const auto result = extractor.extract(document);

        fs::create_directories("resultados");

        const fs::path output =
            fs::path("resultados") / "resultado.json";

        std::ofstream file(output);

        if (!file) {
            throw std::runtime_error(
                "Nao foi possivel criar o JSON."
            );
        }

        file << result.toJson();

        std::cout << "Resultado salvo em: "
                  << output.string() << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
}

// ============================================================
// Programa principal
// ============================================================

int main(int argc, char* argv[]) {
    std::cout << "DocFieldExtractor - C++\n";

    try {
        DocumentExtractor extractor("por+eng");

        // Permite executar um arquivo passado na linha de comando:
        // ./DocFieldExtractor exame.png

        if (argc >= 2) {
            const auto result = extractor.extract(argv[1]);

            result.printSummary();

            if (argc >= 3) {
                std::ofstream output(argv[2]);

                if (!output) {
                    throw std::runtime_error(
                        "Nao foi possivel criar o arquivo de saida."
                    );
                }

                output << result.toJson();

                std::cout << "\nJSON exportado para: "
                          << argv[2] << '\n';
            }

            return 0;
        }

        exampleBasicExtraction(extractor);
        exampleBatchProcessing(extractor);
        exampleJsonExport(extractor);
    }
    catch (const std::exception& error) {
        std::cerr << "\nErro: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
