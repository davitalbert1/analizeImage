
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

namespace fs = std::filesystem;

// ============================================================
// Configuracoes
// ============================================================

constexpr int WIDTH = 800;
constexpr int HEIGHT = 1100;

const cv::Scalar WHITE(255, 255, 255);
const cv::Scalar BLACK(0, 0, 0);
const cv::Scalar NAVY(80, 0, 0);       // BGR
const cv::Scalar GRAY(128, 128, 128);
const cv::Scalar LIGHT_GRAY(238, 238, 238);

// OpenCV utiliza BGR, nao RGB.

// ============================================================
// Funcoes de desenho
// ============================================================

void drawText(
    cv::Mat& image,
    const std::string& text,
    int x,
    int y,
    double scale,
    const cv::Scalar& color,
    int thickness = 1
) {
    cv::putText(
        image,
        text,
        cv::Point(x, y),
        cv::FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv::LINE_AA
    );
}

void drawLine(
    cv::Mat& image,
    int x1,
    int y1,
    int x2,
    int y2,
    const cv::Scalar& color,
    int thickness = 1
) {
    cv::line(
        image,
        cv::Point(x1, y1),
        cv::Point(x2, y2),
        color,
        thickness,
        cv::LINE_AA
    );
}

// ============================================================
// Desenhar cabecalho de tabela
// ============================================================

void drawTableHeader(cv::Mat& image, int& y) {
    drawText(image, "Exame", 60, y, 0.38, GRAY);
    drawText(image, "Resultado", 300, y, 0.38, GRAY);
    drawText(image, "Unidade", 450, y, 0.38, GRAY);
    drawText(image, "Ref.", 580, y, 0.38, GRAY);

    y += 5;

    drawLine(image, 50, y + 15, 750, y + 15, LIGHT_GRAY);

    y += 40;
}

// ============================================================
// Desenhar linhas de exames
// ============================================================

using Exam = std::vector<std::string>;

void drawExams(
    cv::Mat& image,
    int& y,
    const std::vector<Exam>& exams
) {
    for (const auto& exam : exams) {
        if (exam.size() != 4) {
            continue;
        }

        drawText(image, exam[0], 60, y, 0.42, BLACK);
        drawText(image, exam[1], 300, y, 0.42, BLACK);
        drawText(image, exam[2], 450, y, 0.42, BLACK);
        drawText(image, exam[3], 580, y, 0.34, GRAY);

        y += 25;

        drawLine(
            image, 50, y + 2, 750, y + 2,
            LIGHT_GRAY
        );

        y += 8;
    }
}

// ============================================================
// Criar documento de teste
// ============================================================

fs::path createTestDocument() {
    cv::Mat image(
        HEIGHT,
        WIDTH,
        CV_8UC3,
        WHITE
    );

    int y = 55;

    // --------------------------------------------------------
    // Cabecalho
    // --------------------------------------------------------

    drawText(
        image,
        "LABORATORIO SAUDE",
        220, y, 0.85, NAVY, 2
    );

    y += 35;

    drawText(
        image,
        "Resultado de Exames",
        280, y, 0.55, GRAY
    );

    y += 40;

    drawLine(image, 50, y, 750, y, NAVY, 2);

    y += 35;

    // --------------------------------------------------------
    // Dados do paciente
    // --------------------------------------------------------

    drawText(image, "Paciente:", 60, y, 0.42, BLACK);
    drawText(image, "Maria Santos Silva", 200, y, 0.42, BLACK);
    y += 25;

    drawText(image, "Data:", 60, y, 0.42, BLACK);
    drawText(image, "15/10/2026", 200, y, 0.42, BLACK);
    y += 25;

    drawText(image, "Idade:", 60, y, 0.42, BLACK);
    drawText(image, "45 anos", 200, y, 0.42, BLACK);
    y += 25;

    drawText(image, "Medico:", 60, y, 0.42, BLACK);
    drawText(
        image,
        "Dr. Carlos Mendes - CRM 12345",
        200, y, 0.42, BLACK
    );

    y += 35;

    drawLine(image, 50, y, 750, y, GRAY);

    y += 35;

    // --------------------------------------------------------
    // Bioquimica
    // --------------------------------------------------------

    drawText(image, "BIOQUIMICA", 60, y, 0.55, NAVY, 2);

    y += 35;

    drawTableHeader(image, y);

    const std::vector<Exam> exams = {
        {"Glicose", "95", "mg/dL", "70 a 99"},
        {"Colesterol Total", "180", "mg/dL", "< 200"},
        {"HDL Colesterol", "55", "mg/dL", "> 40"},
        {"LDL Colesterol", "105", "mg/dL", "< 130"},
        {"Triglicerideos", "120", "mg/dL", "< 150"},
        {"Ureia", "32", "mg/dL", "15 a 45"},
        {"Creatinina", "0.9", "mg/dL", "0.6 a 1.2"},
        {"Acido Urico", "5.2", "mg/dL", "2.5 a 7.0"},
        {"TGO (AST)", "25", "U/L", "< 40"},
        {"TGP (ALT)", "30", "U/L", "< 41"}
    };

    drawExams(image, y, exams);

    // --------------------------------------------------------
    // Hemograma
    // --------------------------------------------------------

    y += 15;

    drawLine(image, 50, y, 750, y, GRAY);

    y += 35;

    drawText(image, "HEMOGRAMA", 60, y, 0.55, NAVY, 2);

    y += 35;

    drawTableHeader(image, y);

    const std::vector<Exam> hemograma = {
        {"Hemoglobina", "14.2", "g/dL", "12.0 a 16.0"},
        {"Hematocrito", "42.5", "%", "36 a 46"},
        {"Leucocitos", "7.500", "/mm3", "4.000 a 11.000"},
        {"Plaquetas", "250.000", "/mm3", "150.000 a 400.000"},
        {"Hemacia", "4.8", "milhoes/mm3", "4.0 a 5.5"},
        {"VCM", "88", "fL", "80 a 100"}
    };

    drawExams(image, y, hemograma);

    // --------------------------------------------------------
    // Rodape
    // --------------------------------------------------------

    y += 20;

    drawLine(image, 50, y, 750, y, NAVY, 2);

    y += 30;

    drawText(
        image,
        "Responsavel Tecnico: Dra. Ana Oliveira",
        200, y, 0.38, GRAY
    );

    // --------------------------------------------------------
    // Salvar arquivo
    // --------------------------------------------------------

    const fs::path outputPath =
        fs::current_path() /
        "test_docs" /
        "exame_sangue_teste.png";

    fs::create_directories(outputPath.parent_path());

    if (!cv::imwrite(outputPath.string(), image)) {
        throw std::runtime_error(
            "Nao foi possivel salvar o documento PNG."
        );
    }

    std::cout << "Documento de teste criado:\n"
              << outputPath.string() << '\n';

    return outputPath;
}

// ============================================================
// Main
// ============================================================

int main() {
    try {
        createTestDocument();
    }
    catch (const std::exception& error) {
        std::cerr << "Erro: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
