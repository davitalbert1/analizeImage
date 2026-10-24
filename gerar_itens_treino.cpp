#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <locale>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

namespace fs = std::filesystem;
using nlohmann::json;

// ---------------------------------------------------------------------------
// Cores e estilos
// ---------------------------------------------------------------------------

struct Rgb {
    int r = 0;
    int g = 0;
    int b = 0;
};

cv::Scalar paraBgr(const Rgb& cor) {
    return cv::Scalar(
        static_cast<double>(cor.b),
        static_cast<double>(cor.g),
        static_cast<double>(cor.r)
    );
}

Rgb escurecer(const Rgb& cor, int delta) {
    return {
        std::max(0, cor.r - delta),
        std::max(0, cor.g - delta),
        std::max(0, cor.b - delta),
    };
}

Rgb clarear(const Rgb& cor, int delta) {
    return {
        std::min(255, cor.r + delta),
        std::min(255, cor.g + delta),
        std::min(255, cor.b + delta),
    };
}

struct VisualStyle {
    std::string fonte;
    Rgb corTexto;
    Rgb corFundo;
    bool fundoCinza = false;
};

const std::vector<VisualStyle> ESTILOS_VISUAIS = {
    {"arial.ttf", {0, 0, 0}, {255, 255, 255}, false},
    {"arialbd.ttf", {20, 20, 60}, {255, 255, 255}, false},
    {"calibri.ttf", {30, 30, 30}, {245, 245, 240}, true},
    {"times.ttf", {0, 0, 0}, {255, 255, 255}, false},
    {"verdana.ttf", {40, 40, 40}, {250, 250, 250}, true},
    {"consola.ttf", {0, 40, 0}, {255, 255, 255}, false},
    {"cour.ttf", {60, 0, 0}, {255, 252, 240}, false},
};

const std::vector<std::string> FONTES_FALLBACK = {
    "arial.ttf", "calibri.ttf", "segoeui.ttf", "times.ttf",
    "verdana.ttf", "DejaVuSans.ttf", "DejaVuSansMono.ttf",
};

const std::vector<Rgb> CORES_TITULO = {
    {0, 0, 0}, {20, 40, 90}, {90, 20, 20}, {25, 70, 40}, {60, 30, 80},
};

const std::vector<std::string> LAYOUTS = {
    "lista", "lista_alternada", "duas_colunas", "tabela", "blocos",
};

// ---------------------------------------------------------------------------
// Vocabulário
// ---------------------------------------------------------------------------

const std::vector<std::string> NOMES = {
    "Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela",
    "Henrique", "Isabela", "João", "Karina", "Lucas", "Mariana",
    "Nicolas", "Olívia", "Pedro", "Rafaela", "Samuel", "Tatiane",
    "Victor", "Wagner", "Yasmin", "Zeca",
};

const std::vector<std::string> SOBRENOMES = {
    "Silva", "Santos", "Oliveira", "Souza", "Pereira", "Lima",
    "Ferreira", "Almeida", "Costa", "Rodrigues", "Gomes", "Martins",
    "Rocha", "Ribeiro", "Carvalho", "Mendes", "Dias", "Nunes",
    "Farias", "Barros", "Castro", "Moraes", "Teixeira", "Moreira",
};

const std::vector<std::string> CIDADES = {
    "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador",
    "Brasília", "Fortaleza", "Curitiba", "Manaus", "Recife",
    "Porto Alegre", "Goiânia", "Belém",
};

const std::vector<std::string> ESTADOS = {
    "SP", "RJ", "MG", "BA", "DF", "CE", "PR", "AM", "PE", "RS", "GO", "PA",
};

const std::vector<std::string> GENERICO_TERMOS = {
    "Alpha", "Beta", "Delta", "Omega", "Nova", "Aurora", "Atlas",
    "Bravo", "Cosmos", "Eureka", "Fenix", "Horizonte", "Íris", "Lumen",
    "Nébula", "Orion", "Prisma", "Quantum", "Rubi", "Solar", "Titan",
    "Ultra", "Vertex", "Zenite",
};

const std::vector<std::string> CATEGORIAS = {
    "Serviço", "Produto", "Contrato", "Pedido", "Chamado", "Projeto",
    "Cadastro", "Solicitação", "Registro", "Ordem",
};

const std::vector<std::string> STATUS = {
    "Aberto", "Fechado", "Pendente", "Em análise", "Aprovado",
    "Concluído", "Cancelado", "Suspenso",
};

// ---------------------------------------------------------------------------
// Sorteios
// ---------------------------------------------------------------------------

class Random {
public:
    explicit Random(std::uint32_t semente) : motor_(semente) {}

    int inteiro(int minimo, int maximo) {
        if (maximo < minimo) {
            std::swap(minimo, maximo);
        }
        std::uniform_int_distribution<int> distribuicao(minimo, maximo);
        return distribuicao(motor_);
    }

    double real(double minimo, double maximo) {
        std::uniform_real_distribution<double> distribuicao(minimo, maximo);
        return distribuicao(motor_);
    }

    bool chance(double probabilidade) {
        return std::generate_canonical<double, 53>(motor_) < probabilidade;
    }

    template <typename T>
    const T& escolher(const std::vector<T>& valores) {
        const int indice = inteiro(0, static_cast<int>(valores.size()) - 1);
        return valores[static_cast<std::size_t>(indice)];
    }

    void embaralhar(std::vector<int>& valores) {
        std::shuffle(valores.begin(), valores.end(), motor_);
    }

    std::mt19937& motor() {
        return motor_;
    }

private:
    std::mt19937 motor_;
};

std::string escolha(Random& rng, std::initializer_list<std::string_view> opcoes) {
    auto it = opcoes.begin();
    std::advance(it, rng.inteiro(0, static_cast<int>(opcoes.size()) - 1));
    return std::string(*it);
}

int escolhaInteiro(Random& rng, std::initializer_list<int> opcoes) {
    auto it = opcoes.begin();
    std::advance(it, rng.inteiro(0, static_cast<int>(opcoes.size()) - 1));
    return *it;
}

char letraMaiuscula(Random& rng) {
    return static_cast<char>('A' + rng.inteiro(0, 25));
}

std::string digitos(Random& rng, int quantidade) {
    std::string resultado;
    resultado.reserve(static_cast<std::size_t>(quantidade));
    for (int i = 0; i < quantidade; ++i) {
        resultado.push_back(static_cast<char>('0' + rng.inteiro(0, 9)));
    }
    return resultado;
}

std::string sortearNome(Random& rng) {
    const std::string primeiro = rng.escolher(NOMES);
    const std::string segundo = rng.escolher(SOBRENOMES);
    const std::string terceiro = rng.escolher(SOBRENOMES);
    return primeiro + " " + segundo + " " + terceiro;
}

std::string sortearData(Random& rng, int diasAtras = 730) {
    const std::time_t agora = std::time(nullptr);
    std::tm base = *std::localtime(&agora);
    base.tm_hour = 12;
    base.tm_min = 0;
    base.tm_sec = 0;
    std::time_t inicial = std::mktime(&base);
    inicial -= static_cast<std::time_t>(diasAtras) * 24 * 60 * 60;
    const int deslocamento = rng.inteiro(0, std::max(diasAtras, 0));
    const std::time_t sorteada = inicial + static_cast<std::time_t>(deslocamento) * 24 * 60 * 60;
    const std::tm resultado = *std::localtime(&sorteada);
    char buffer[16] = {};
    std::strftime(buffer, sizeof(buffer), "%d/%m/%Y", &resultado);
    return std::string(buffer);
}

std::string formatarDecimal(double valor, int casas) {
    std::ostringstream saida;
    saida.imbue(std::locale::classic());
    saida << std::fixed << std::setprecision(casas) << valor;
    return saida.str();
}

std::string aparar(const std::string& texto) {
    const auto inicio = texto.find_first_not_of(" \t\r\n");
    if (inicio == std::string::npos) {
        return "";
    }
    const auto fim = texto.find_last_not_of(" \t\r\n");
    return texto.substr(inicio, fim - inicio + 1);
}

std::vector<std::string> dividir(const std::string& texto, char separador) {
    std::vector<std::string> partes;
    std::string parte;
    std::istringstream entrada(texto);
    while (std::getline(entrada, parte, separador)) {
        partes.push_back(parte);
    }
    return partes;
}

std::string juntar(const std::vector<std::string>& itens, const std::string& separador) {
    std::ostringstream saida;
    for (std::size_t i = 0; i < itens.size(); ++i) {
        if (i > 0) {
            saida << separador;
        }
        saida << itens[i];
    }
    return saida.str();
}

std::string formatarDuracao(double segundos) {
    if (segundos < 0.0 || !std::isfinite(segundos)) {
        return "--:--:--";
    }
    const long long total = static_cast<long long>(std::llround(segundos));
    const long long horas = total / 3600;
    const long long minutos = (total % 3600) / 60;
    const long long resto = total % 60;
    std::ostringstream saida;
    saida << std::setfill('0') << std::setw(2) << horas
          << ':' << std::setw(2) << minutos
          << ':' << std::setw(2) << resto;
    return saida.str();
}

// ---------------------------------------------------------------------------
// Perfis de documento
// ---------------------------------------------------------------------------

using Generator = std::function<std::string(Random&)>;

struct FieldDef {
    std::string rotulo;
    Generator gerar;
};

struct DocProfile {
    std::string chave;
    std::vector<std::string> titulos;
    std::vector<std::string> subtitulos;
    std::vector<FieldDef> fields;
};

using FieldList = std::vector<std::pair<std::string, std::string>>;
using Registro = std::map<std::string, std::string>;

std::vector<FieldDef> criarCamposComuns() {
    return {
        {"Código", [](Random& rng) {
            const char letra = letraMaiuscula(rng);
            const int parte1 = rng.inteiro(1000, 9999);
            const int parte2 = rng.inteiro(10, 99);
            return std::string(1, letra) + std::to_string(parte1) + "-" + std::to_string(parte2);
        }},
        {"Data", [](Random& rng) {
            return sortearData(rng);
        }},
        {"Nome", [](Random& rng) {
            return sortearNome(rng);
        }},
        {"Identificador", [](Random& rng) {
            return digitos(rng, 9);
        }},
        {"Categoria", [](Random& rng) {
            return rng.escolher(CATEGORIAS);
        }},
        {"Status", [](Random& rng) {
            return rng.escolher(STATUS);
        }},
        {"Responsável", [](Random& rng) {
            return sortearNome(rng);
        }},
        {"Observação", [](Random& rng) {
            return escolha(rng, {
                "Sem observações adicionais.",
                "Verificar pendências.",
                "Processado conforme solicitação.",
                "Aguardando aprovação.",
                "Documento conferido.",
                "Sujeito a conferência posterior.",
            });
        }},
        {"Valor", [](Random& rng) {
            return formatarDecimal(rng.real(10.0, 9999.0), 2);
        }},
        {"Quantidade", [](Random& rng) {
            return std::to_string(rng.inteiro(1, 500));
        }},
        {"Cidade", [](Random& rng) {
            return rng.escolher(CIDADES);
        }},
        {"Estado", [](Random& rng) {
            return rng.escolher(ESTADOS);
        }},
        {"Referência", [](Random& rng) {
            return "REF-" + std::to_string(rng.inteiro(10000, 99999));
        }},
    };
}

const std::vector<FieldDef>& camposComuns() {
    static const std::vector<FieldDef> campos = criarCamposComuns();
    return campos;
}

std::vector<DocProfile> criarPerfis() {
    std::vector<DocProfile> perfis;

    perfis.push_back({
        "financeiro",
        {
            "RELATÓRIO FINANCEIRO", "COMPROVANTE DE PAGAMENTO",
            "EXTRATO DE CONTA", "DEMONSTRATIVO DE VALORES",
            "NOTA DE DÉBITO", "CONTROLE DE PAGAMENTOS",
        },
        {"Período de referência", "Consolidado", "Detalhado"},
        {
            {"Descrição", [](Random& rng) {
                return escolha(rng, {
                    "Transferência entre contas", "Pagamento de fornecedor",
                    "Crédito de operação", "Ajuste de saldo",
                    "Despesa administrativa", "Repasse de valores",
                });
            }},
            {"Valor", [](Random& rng) {
                return "R$ " + formatarDecimal(rng.real(10.0, 50000.0), 2);
            }},
            {"Moeda", [](Random&) {
                return std::string("BRL");
            }},
            {"Centro de Custo", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Conta", [](Random& rng) {
                const int parte1 = rng.inteiro(1000, 9999);
                const int parte2 = rng.inteiro(10, 99);
                return std::to_string(parte1) + "-" + std::to_string(parte2);
            }},
            {"Banco", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Tipo de Operação", [](Random& rng) {
                return escolha(rng, {
                    "Crédito", "Débito", "Transferência", "Pagamento",
                });
            }},
            {"Data de Vencimento", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Valor Total", [](Random& rng) {
                return "R$ " + formatarDecimal(rng.real(100.0, 100000.0), 2);
            }},
            {"Pago", [](Random& rng) {
                return escolha(rng, {"Sim", "Não", "Parcial"});
            }},
        },
    });

    perfis.push_back({
        "transporte",
        {
            "COMPROVANTE DE TRANSPORTE", "NOTA DE REMESSA",
            "RELATÓRIO DE ROTA", "TERMO DE ENTREGA",
            "REGISTRO DE DESPACHO", "ACOMPANHAMENTO DE CARGA",
        },
        {"Origem", "Destino", "Rota"},
        {
            {"Origem", [](Random& rng) {
                return rng.escolher(CIDADES);
            }},
            {"Destino", [](Random& rng) {
                return rng.escolher(CIDADES);
            }},
            {"Motorista", [](Random& rng) {
                return sortearNome(rng);
            }},
            {"Placa", [](Random& rng) {
                std::string placa;
                for (int i = 0; i < 3; ++i) {
                    placa.push_back(letraMaiuscula(rng));
                }
                for (int i = 0; i < 4; ++i) {
                    placa.push_back(static_cast<char>('0' + rng.inteiro(0, 9)));
                }
                return placa;
            }},
            {"Distância", [](Random& rng) {
                return std::to_string(rng.inteiro(5, 1200)) + " km";
            }},
            {"Peso", [](Random& rng) {
                return formatarDecimal(rng.real(1.0, 5000.0), 1) + " kg";
            }},
            {"Data de Coleta", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Data de Entrega", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Tipo de Frete", [](Random& rng) {
                return escolha(rng, {
                    "Normal", "Expresso", "Programado", "Especial",
                });
            }},
            {"Cidade", [](Random& rng) {
                return rng.escolher(CIDADES);
            }},
        },
    });

    perfis.push_back({
        "recursos_humanos",
        {
            "REGISTRO DE COLABORADOR", "FORMULÁRIO DE ADMISSÃO",
            "SOLICITAÇÃO DE FÉRIAS", "COMPROVANTE DE TREINAMENTO",
            "FICHA DE CADASTRO", "AVALIAÇÃO DE DESEMPENHO",
        },
        {"Colaborador", "Cadastro", "Movimentação"},
        {
            {"Colaborador", [](Random& rng) {
                return sortearNome(rng);
            }},
            {"Cargo", [](Random& rng) {
                return escolha(rng, {
                    "Analista", "Técnico", "Coordenador", "Assistente",
                    "Gerente", "Especialista", "Operacional",
                });
            }},
            {"Departamento", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Data de Admissão", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Tipo de Contrato", [](Random& rng) {
                return escolha(rng, {
                    "CLT", "PJ", "Estágio", "Temporário",
                });
            }},
            {"Carga Horária", [](Random& rng) {
                return escolha(rng, {
                    "40h semanais", "44h semanais", "30h semanais", "Escala 12x36",
                });
            }},
            {"Status", [](Random& rng) {
                return rng.escolher(STATUS);
            }},
            {"Observação", [](Random& rng) {
                return escolha(rng, {
                    "Sem observações adicionais.",
                    "Documentação pendente.",
                    "Processado conforme solicitação.",
                    "Aguardando aprovação.",
                });
            }},
            {"Identificador", [](Random& rng) {
                return digitos(rng, 9);
            }},
        },
    });

    perfis.push_back({
        "saude",
        {
            "RELATÓRIO DE EXAMES", "RESULTADOS CLÍNICOS",
            "LAUDO MÉDICO", "EXAME LABORATORIAL",
            "COMPROVANTE DE ATENDIMENTO", "FICHA DE ACOMPANHAMENTO",
        },
        {"Paciente", "Atendimento", "Laudo"},
        {
            {"Paciente", [](Random& rng) {
                return sortearNome(rng);
            }},
            {"Profissional", [](Random& rng) {
                return sortearNome(rng);
            }},
            {"Data do Exame", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Tipo de Exame", [](Random& rng) {
                return escolha(rng, {
                    "Laboratorial", "Imagem", "Clínico", "Funcional",
                });
            }},
            {"Unidade", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Resultado", [](Random& rng) {
                return escolha(rng, {
                    "Dentro do esperado", "Alterado", "Em observação",
                    "Normal", "Acompanhamento necessário",
                });
            }},
            {"Observação", [](Random& rng) {
                return escolha(rng, {
                    "Sem observações adicionais.",
                    "Repetir avaliação em 90 dias.",
                    "Encaminhado para acompanhamento.",
                    "Documento conferido.",
                });
            }},
            {"Valor", [](Random& rng) {
                return formatarDecimal(rng.real(10.0, 900.0), 2);
            }},
        },
    });

    perfis.push_back({
        "logistica",
        {
            "NOTA DE EMBARQUE", "CONTROLE DE ARMAZÉM",
            "INVENTÁRIO DE ESTOQUE", "REGISTRO DE EXPEDIÇÃO",
            "PEDIDO DE COMPRA", "CONFERÊNCIA DE CARGA",
        },
        {"Armazém", "Expedição", "Estoque"},
        {
            {"Produto", [](Random& rng) {
                const std::string termo = rng.escolher(GENERICO_TERMOS);
                return termo + " " + std::to_string(rng.inteiro(1, 999));
            }},
            {"SKU", [](Random& rng) {
                static const std::string caracteres =
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
                std::string sku;
                for (int i = 0; i < 8; ++i) {
                    sku.push_back(caracteres[static_cast<std::size_t>(rng.inteiro(0, static_cast<int>(caracteres.size()) - 1))]);
                }
                return sku;
            }},
            {"Quantidade", [](Random& rng) {
                return std::to_string(rng.inteiro(1, 5000));
            }},
            {"Armazém", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Setor", [](Random& rng) {
                const char letra = letraMaiuscula(rng);
                return std::string(1, letra) + std::to_string(rng.inteiro(1, 20));
            }},
            {"Data de Entrada", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Data de Saída", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Fornecedor", [](Random& rng) {
                return sortearNome(rng);
            }},
            {"Valor Unitário", [](Random& rng) {
                return "R$ " + formatarDecimal(rng.real(1.0, 5000.0), 2);
            }},
            {"Status", [](Random& rng) {
                return rng.escolher(STATUS);
            }},
        },
    });

    perfis.push_back({
        "energia",
        {
            "FATURA DE ENERGIA", "RELATÓRIO DE CONSUMO",
            "MEDIÇÃO DE UNIDADE", "DETALHAMENTO DE TARIFA",
            "HISTÓRICO DE CONSUMO", "COMPROVANTE DE LEITURA",
        },
        {"Consumo", "Tarifa", "Leitura"},
        {
            {"Unidade Consumidora", [](Random& rng) {
                return std::to_string(rng.inteiro(10000000, 99999999));
            }},
            {"Consumo", [](Random& rng) {
                return std::to_string(rng.inteiro(20, 5000)) + " kWh";
            }},
            {"Leitura Anterior", [](Random& rng) {
                return std::to_string(rng.inteiro(1000, 99999));
            }},
            {"Leitura Atual", [](Random& rng) {
                return std::to_string(rng.inteiro(10000, 199999));
            }},
            {"Tarifa", [](Random& rng) {
                return escolha(rng, {"B1", "B2", "B3", "A1", "A2", "G1"});
            }},
            {"Vencimento", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Valor", [](Random& rng) {
                return "R$ " + formatarDecimal(rng.real(50.0, 3000.0), 2);
            }},
            {"Bandeira", [](Random& rng) {
                return escolha(rng, {"Verde", "Amarela", "Vermelha"});
            }},
        },
    });

    perfis.push_back({
        "telecom",
        {
            "FATURA DE SERVIÇO", "RELATÓRIO DE ASSINATURA",
            "DETALHAMENTO DE USO", "COMPROVANTE DE INSTALAÇÃO",
            "SOLICITAÇÃO DE PLANO", "CONTROLE DE CONTRATO",
        },
        {"Assinante", "Plano", "Uso"},
        {
            {"Assinante", [](Random& rng) {
                return sortearNome(rng);
            }},
            {"Linha", [](Random& rng) {
                const int ddd = rng.inteiro(11, 99);
                const int prefixo = rng.inteiro(90000, 99999);
                const int sufixo = rng.inteiro(1000, 9999);
                return "(" + std::to_string(ddd) + ") " +
                       std::to_string(prefixo) + "-" + std::to_string(sufixo);
            }},
            {"Plano", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Data de Ativação", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Valor da Mensalidade", [](Random& rng) {
                return "R$ " + formatarDecimal(rng.real(29.0, 300.0), 2);
            }},
            {"Data de Vencimento", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Status", [](Random& rng) {
                return rng.escolher(STATUS);
            }},
            {"Observação", [](Random& rng) {
                return escolha(rng, {
                    "Sem observações adicionais.",
                    "Ajuste de titularidade solicitado.",
                    "Aguardando instalação.",
                    "Documento conferido.",
                });
            }},
        },
    });

    perfis.push_back({
        "seguranca",
        {
            "RELATÓRIO DE OCORRÊNCIA", "REGISTRO DE OCORRÊNCIA",
            "COMPROVANTE DE APÓLICE", "RELATÓRIO DE PATRULHA",
            "REGISTRO DE VISITA", "CONTROLE DE RISCO",
        },
        {"Ocorrência", "Apólice", "Vistoria"},
        {
            {"Segurado", [](Random& rng) {
                return sortearNome(rng);
            }},
            {"Número da Apólice", [](Random& rng) {
                return std::to_string(rng.inteiro(100000, 999999));
            }},
            {"Tipo de Cobertura", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Vigência", [](Random& rng) {
                const std::string inicio = sortearData(rng);
                const std::string fim = sortearData(rng);
                return inicio + " a " + fim;
            }},
            {"Valor da Apólice", [](Random& rng) {
                return "R$ " + formatarDecimal(rng.real(100.0, 20000.0), 2);
            }},
            {"Ocorrido em", [](Random& rng) {
                return sortearData(rng);
            }},
            {"Cidade", [](Random& rng) {
                return rng.escolher(CIDADES);
            }},
            {"Estado", [](Random& rng) {
                return rng.escolher(ESTADOS);
            }},
            {"Status", [](Random& rng) {
                return rng.escolher(STATUS);
            }},
        },
    });

    perfis.push_back({
        "academico",
        {
            "DECLARAÇÃO DE MATRÍCULA", "COMPROVANTE DE NOTAS",
            "DECLARAÇÃO DE FREQUÊNCIA", "HISTÓRICO ESCOLAR",
            "CERTIFICADO DE CONCLUSÃO", "FICHA DE ALUNO",
        },
        {"Aluno", "Disciplina", "Período"},
        {
            {"Aluno", [](Random& rng) {
                return sortearNome(rng);
            }},
            {"Matrícula", [](Random& rng) {
                return digitos(rng, 8);
            }},
            {"Curso", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Período", [](Random& rng) {
                const int ano = rng.inteiro(2018, 2026);
                const int semestre = rng.inteiro(1, 2);
                return std::to_string(ano) + "." + std::to_string(semestre);
            }},
            {"Disciplina", [](Random& rng) {
                return rng.escolher(GENERICO_TERMOS);
            }},
            {"Nota", [](Random& rng) {
                return formatarDecimal(rng.real(0.0, 10.0), 1);
            }},
            {"Frequência", [](Random& rng) {
                return std::to_string(rng.inteiro(60, 100)) + "%";
            }},
            {"Situação", [](Random& rng) {
                return escolha(rng, {"Aprovado", "Reprovado", "Em andamento"});
            }},
            {"Data de Emissão", [](Random& rng) {
                return sortearData(rng);
            }},
        },
    });

    perfis.push_back({
        "generico",
        {
            "DOCUMENTO", "REGISTRO GERAL", "FORMULÁRIO",
            "COMPROVANTE", "RELATÓRIO", "FICHA DE REGISTRO",
        },
        {"Registro", "Geral", "Controle"},
        camposComuns(),
    });

    return perfis;
}

const std::vector<DocProfile>& perfisDocumento() {
    static const std::vector<DocProfile> perfis = criarPerfis();
    return perfis;
}

const DocProfile& perfilPorNome(const std::string& nome) {
    for (const auto& perfil : perfisDocumento()) {
        if (perfil.chave == nome) {
            return perfil;
        }
    }
    throw std::runtime_error("Tipo de documento desconhecido: " + nome);
}

bool tipoExiste(const std::string& nome) {
    for (const auto& perfil : perfisDocumento()) {
        if (perfil.chave == nome) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> nomesDosTipos() {
    std::vector<std::string> nomes;
    nomes.reserve(perfisDocumento().size());
    for (const auto& perfil : perfisDocumento()) {
        nomes.push_back(perfil.chave);
    }
    return nomes;
}

bool layoutExiste(const std::string& layout) {
    return std::find(LAYOUTS.begin(), LAYOUTS.end(), layout) != LAYOUTS.end();
}

const FieldDef& campoPorNome(const std::vector<FieldDef>& campos, const std::string& nome) {
    for (const auto& campo : campos) {
        if (campo.rotulo == nome) {
            return campo;
        }
    }
    throw std::runtime_error("Campo desconhecido: " + nome);
}

bool temCampo(const std::vector<FieldDef>& campos, const std::string& nome) {
    for (const auto& campo : campos) {
        if (campo.rotulo == nome) {
            return true;
        }
    }
    return false;
}

FieldList gerarCamposAleatorios(Random& rng, const std::string& tipo) {
    const DocProfile& perfil = perfilPorNome(tipo);
    std::vector<FieldDef> pool = perfil.fields;

    static const std::vector<std::string> camposGarantidos = {
        "Código", "Data", "Nome", "Identificador",
    };
    for (const auto& chave : camposGarantidos) {
        if (!temCampo(pool, chave)) {
            pool.push_back(campoPorNome(camposComuns(), chave));
        }
    }

    const int maximo = std::min(11, static_cast<int>(pool.size()));
    const int minimo = std::min(6, maximo);
    const int quantidade = rng.inteiro(minimo, maximo);

    std::vector<int> indices(pool.size());
    std::iota(indices.begin(), indices.end(), 0);
    rng.embaralhar(indices);

    std::vector<std::string> chaves;
    chaves.reserve(static_cast<std::size_t>(quantidade));
    for (int i = 0; i < quantidade; ++i) {
        chaves.push_back(pool[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])].rotulo);
    }

    static const std::vector<std::string> preferidos = {"Código", "Data"};
    for (const auto& obrigatorio : preferidos) {
        const bool jaSelecionado =
            std::find(chaves.begin(), chaves.end(), obrigatorio) != chaves.end();
        if (temCampo(pool, obrigatorio) && !jaSelecionado && rng.chance(0.75) && !chaves.empty()) {
            chaves.front() = obrigatorio;
        }
    }

    FieldList campos;
    campos.reserve(chaves.size());
    for (const auto& chave : chaves) {
        std::string valor;
        try {
            valor = campoPorNome(pool, chave).gerar(rng);
        } catch (const std::exception&) {
            valor.clear();
        }
        campos.emplace_back(chave, valor);
    }
    return campos;
}

// ---------------------------------------------------------------------------
// UTF-8 e fontes
// ---------------------------------------------------------------------------

std::size_t contarCodepoints(const std::string& texto) {
    return static_cast<std::size_t>(std::count_if(
        texto.begin(), texto.end(),
        [](unsigned char c) { return (c & 0xC0) != 0x80; }
    ));
}

void removerUltimoCodepoint(std::string& texto) {
    if (texto.empty()) {
        return;
    }
    std::size_t inicio = texto.size() - 1;
    while (inicio > 0 && (static_cast<unsigned char>(texto[inicio]) & 0xC0) == 0x80) {
        --inicio;
    }
    texto.erase(inicio);
}

bool iguaisSemCaixa(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    return std::equal(
        a.begin(), a.end(), b.begin(),
        [](unsigned char x, unsigned char y) {
            return std::tolower(x) == std::tolower(y);
        }
    );
}

std::vector<fs::path> diretoriosDeFontes() {
    std::vector<fs::path> diretorios;
    diretorios.emplace_back("fonts");
    if (const char* windir = std::getenv("WINDIR")) {
        diretorios.emplace_back(fs::path(windir) / "Fonts");
    }
    if (const char* systemroot = std::getenv("SystemRoot")) {
        diretorios.emplace_back(fs::path(systemroot) / "Fonts");
    }
    diretorios.emplace_back("C:/Windows/Fonts");
    if (const char* localappdata = std::getenv("LOCALAPPDATA")) {
        diretorios.emplace_back(fs::path(localappdata) / "Microsoft" / "Windows" / "Fonts");
    }
    diretorios.emplace_back("/usr/share/fonts/truetype/dejavu");
    diretorios.emplace_back("/usr/share/fonts/truetype/msttcorefonts");
    diretorios.emplace_back("/usr/share/fonts/truetype/liberation");
    diretorios.emplace_back("/usr/share/fonts/truetype");
    diretorios.emplace_back("/usr/share/fonts/TTF");
    diretorios.emplace_back("/usr/share/fonts");
    diretorios.emplace_back("/usr/local/share/fonts");
    if (const char* home = std::getenv("HOME")) {
        diretorios.emplace_back(fs::path(home) / ".fonts");
        diretorios.emplace_back(fs::path(home) / ".local" / "share" / "fonts");
    }
    diretorios.emplace_back("/System/Library/Fonts");
    diretorios.emplace_back("/Library/Fonts");
    return diretorios;
}

class TextRenderer {
public:
    int largura(const std::string& nomeFonte, const std::string& texto, int tamanho) {
        if (texto.empty()) {
            return 0;
        }
        int linhaBase = 0;
        return obterFonte(nomeFonte)->getTextSize(texto, tamanho, -1, &linhaBase).width;
    }

    void desenhar(
        cv::Mat& imagem,
        const std::string& nomeFonte,
        const std::string& texto,
        int x,
        int y,
        int tamanho,
        const Rgb& cor
    ) {
        if (texto.empty()) {
            return;
        }
        obterFonte(nomeFonte)->putText(
            imagem,
            texto,
            cv::Point(x, y),
            tamanho,
            paraBgr(cor),
            -1,
            cv::LINE_AA,
            false
        );
    }

private:
    std::map<std::string, std::string> caminhos_;
    std::map<std::string, cv::Ptr<cv::freetype::FreeType2>> fontes_;

    std::string resolverFonte(const std::string& nome) {
        const auto cache = caminhos_.find(nome);
        if (cache != caminhos_.end()) {
            return cache->second;
        }

        const fs::path caminhoDireto(nome);
        std::error_code erro;
        if (caminhoDireto.has_parent_path() && fs::is_regular_file(caminhoDireto, erro)) {
            const std::string caminho = caminhoDireto.string();
            caminhos_[nome] = caminho;
            return caminho;
        }

        std::vector<std::string> candidatos{nome};
        for (const auto& fallback : FONTES_FALLBACK) {
            if (fallback != nome) {
                candidatos.push_back(fallback);
            }
        }

        static const std::vector<fs::path> diretorios = diretoriosDeFontes();

        for (const auto& candidato : candidatos) {
            for (const auto& diretorio : diretorios) {
                const fs::path caminho = diretorio / candidato;
                erro.clear();
                if (fs::is_regular_file(caminho, erro)) {
                    const std::string encontrado = caminho.string();
                    caminhos_[nome] = encontrado;
                    return encontrado;
                }
            }
        }

        for (const auto& candidato : candidatos) {
            for (const auto& diretorio : diretorios) {
                erro.clear();
                if (!fs::is_directory(diretorio, erro)) {
                    continue;
                }
                try {
                    for (fs::recursive_directory_iterator it(
                             diretorio,
                             fs::directory_options::skip_permission_denied
                         ),
                         fim;
                         it != fim;
                         ++it) {
                        std::error_code erroArquivo;
                        if (!it->is_regular_file(erroArquivo)) {
                            continue;
                        }
                        if (iguaisSemCaixa(it->path().filename().string(), candidato)) {
                            const std::string encontrado = it->path().string();
                            caminhos_[nome] = encontrado;
                            return encontrado;
                        }
                    }
                } catch (const fs::filesystem_error&) {
                }
            }
        }

        throw std::runtime_error(
            "Nenhuma fonte TrueType encontrada para '" + nome +
            "'. Copie um arquivo .ttf para ./fonts ou instale fontes no sistema."
        );
    }

    cv::Ptr<cv::freetype::FreeType2> obterFonte(const std::string& nome) {
        const std::string caminho = resolverFonte(nome);
        const auto existente = fontes_.find(caminho);
        if (existente != fontes_.end()) {
            return existente->second;
        }
        cv::Ptr<cv::freetype::FreeType2> fonte = cv::freetype::createFreeType2();
        fonte->loadFontData(caminho, 0);
        fontes_[caminho] = fonte;
        return fonte;
    }
};

std::string truncar(
    TextRenderer& texto,
    const std::string& fonte,
    const std::string& valor,
    int tamanho,
    int larguraMaxima
) {
    std::string resultado = valor;
    while (texto.largura(fonte, resultado, tamanho) > larguraMaxima &&
           contarCodepoints(resultado) > 4) {
        removerUltimoCodepoint(resultado);
        removerUltimoCodepoint(resultado);
    }
    return resultado;
}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

int renderizarLista(
    TextRenderer& texto,
    cv::Mat& imagem,
    const FieldList& campos,
    int x,
    int y,
    int larguraUtil,
    const std::string& fonte,
    int tamanhoFonte,
    const Rgb& corTexto,
    const Rgb&,
    int alturaLinha,
    Registro& registro
) {
    for (const auto& [chave, valor] : campos) {
        const std::string rotulo = chave + ":";
        texto.desenhar(imagem, fonte, rotulo, x, y, tamanhoFonte, corTexto);
        const int larguraRotulo = texto.largura(fonte, rotulo, tamanhoFonte);
        if (!valor.empty()) {
            const std::string valorTexto = truncar(
                texto, fonte, valor, tamanhoFonte,
                larguraUtil - larguraRotulo - 12
            );
            texto.desenhar(
                imagem, fonte, valorTexto,
                x + larguraRotulo + 8, y,
                tamanhoFonte, corTexto
            );
            registro[chave] = valorTexto;
        }
        y += alturaLinha;
    }
    return y;
}

int renderizarListaAlternada(
    TextRenderer& texto,
    cv::Mat& imagem,
    const FieldList& campos,
    int x,
    int y,
    int larguraUtil,
    const std::string& fonte,
    int tamanhoFonte,
    const Rgb& corTexto,
    const Rgb&,
    int alturaLinha,
    const Rgb& corFundoAlternado,
    Registro& registro
) {
    int posicao = 0;
    for (const auto& [chave, valor] : campos) {
        if (posicao % 2 == 1) {
            cv::rectangle(
                imagem,
                cv::Point(x, y - 4),
                cv::Point(x + larguraUtil, y + alturaLinha - 6),
                paraBgr(corFundoAlternado),
                cv::FILLED
            );
        }

        const std::string rotulo = chave + ":";
        texto.desenhar(imagem, fonte, rotulo, x, y, tamanhoFonte, corTexto);
        const int larguraRotulo = texto.largura(fonte, rotulo, tamanhoFonte);
        if (!valor.empty()) {
            const std::string valorTexto = truncar(
                texto, fonte, valor, tamanhoFonte,
                larguraUtil - larguraRotulo - 20
            );
            texto.desenhar(
                imagem, fonte, valorTexto,
                x + larguraRotulo + 8, y,
                tamanhoFonte, corTexto
            );
            registro[chave] = valorTexto;
        }
        y += alturaLinha;
        ++posicao;
    }
    return y;
}

int renderizarBlocos(
    TextRenderer& texto,
    cv::Mat& imagem,
    const FieldList& campos,
    int x,
    int y,
    int larguraUtil,
    const std::string& fonte,
    int tamanhoFonte,
    const Rgb& corTexto,
    const Rgb& corLinha,
    int alturaLinha,
    Registro& registro
) {
    const int larguraCaixa = (larguraUtil - 20) / 2;
    const int alturaCaixa = alturaLinha + 18;

    for (std::size_t i = 0; i < campos.size(); ++i) {
        const auto& [chave, valor] = campos[i];
        const int coluna = static_cast<int>(i % 2);
        const int linha = static_cast<int>(i / 2);
        const int cx = x + coluna * (larguraCaixa + 20);
        const int cy = y + linha * (alturaCaixa + 12);

        cv::rectangle(
            imagem,
            cv::Point(cx, cy),
            cv::Point(cx + larguraCaixa, cy + alturaCaixa),
            paraBgr(corLinha),
            1,
            cv::LINE_AA
        );
        texto.desenhar(imagem, fonte, chave + ":", cx + 10, cy + 8, tamanhoFonte, corTexto);
        if (!valor.empty()) {
            const std::string valorTexto = truncar(
                texto, fonte, valor, tamanhoFonte, larguraCaixa - 24
            );
            texto.desenhar(
                imagem, fonte, valorTexto,
                cx + 10, cy + 8 + alturaLinha - 4,
                tamanhoFonte, corTexto
            );
            registro[chave] = valorTexto;
        }
    }

    const int linhas = (static_cast<int>(campos.size()) + 1) / 2;
    return y + linhas * (alturaCaixa + 12);
}

int renderizarDuasColunas(
    TextRenderer& texto,
    cv::Mat& imagem,
    const FieldList& campos,
    int x,
    int y,
    int larguraUtil,
    const std::string& fonte,
    int tamanhoFonte,
    const Rgb& corTexto,
    const Rgb& corLinha,
    int alturaLinha,
    Registro& registro
) {
    const std::size_t metade = (campos.size() + 1) / 2;
    const FieldList coluna0(
        campos.begin(),
        campos.begin() + static_cast<std::ptrdiff_t>(metade)
    );
    const FieldList coluna1(
        campos.begin() + static_cast<std::ptrdiff_t>(metade),
        campos.end()
    );
    const std::array<const FieldList*, 2> colunas = {&coluna0, &coluna1};
    const int maxLinhas = static_cast<int>(
        std::max(coluna0.size(), coluna1.size())
    );
    const int larguraColuna = (larguraUtil - 30) / 2;

    for (int indiceColuna = 0; indiceColuna < 2; ++indiceColuna) {
        const FieldList& coluna = *colunas[static_cast<std::size_t>(indiceColuna)];
        const int cx = x + indiceColuna * (larguraColuna + 30);
        int cy = y;

        for (const auto& [chave, valor] : coluna) {
            const std::string rotulo = chave + ":";
            texto.desenhar(imagem, fonte, rotulo, cx, cy, tamanhoFonte, corTexto);
            const int larguraRotulo = texto.largura(fonte, rotulo, tamanhoFonte);
            if (!valor.empty()) {
                const std::string valorTexto = truncar(
                    texto, fonte, valor, tamanhoFonte,
                    larguraColuna - larguraRotulo - 12
                );
                texto.desenhar(
                    imagem, fonte, valorTexto,
                    cx + larguraRotulo + 8, cy,
                    tamanhoFonte, corTexto
                );
                registro[chave] = valorTexto;
            }
            cy += alturaLinha;
        }

        if (indiceColuna == 0) {
            const int separadorX = x + larguraColuna + 15;
            cv::line(
                imagem,
                cv::Point(separadorX, y - 5),
                cv::Point(separadorX, y + maxLinhas * alturaLinha),
                paraBgr(corLinha),
                1,
                cv::LINE_AA
            );
        }
    }

    return y + maxLinhas * alturaLinha;
}

int renderizarTabela(
    TextRenderer& texto,
    cv::Mat& imagem,
    const FieldList& campos,
    int x,
    int y,
    int larguraUtil,
    const std::string& fonte,
    int tamanhoFonte,
    const Rgb& corTexto,
    const Rgb& corLinha,
    int alturaLinha,
    Registro& registro
) {
    int larguraMaximaRotulo = 80;
    for (const auto& [chave, valor] : campos) {
        larguraMaximaRotulo = std::max(
            larguraMaximaRotulo,
            texto.largura(fonte, chave + ":", tamanhoFonte)
        );
    }
    const int larguraRotulo = std::min(larguraMaximaRotulo + 16, larguraUtil / 2);
    const int larguraValor = larguraUtil - larguraRotulo;
    const int yInicio = y;
    int indice = 0;

    for (const auto& [chave, valor] : campos) {
        if (indice % 2 == 1) {
            cv::rectangle(
                imagem,
                cv::Point(x, y - 4),
                cv::Point(x + larguraUtil, y + alturaLinha - 6),
                cv::Scalar(238, 238, 238),
                cv::FILLED
            );
        }

        texto.desenhar(imagem, fonte, chave + ":", x + 6, y, tamanhoFonte, corTexto);
        if (!valor.empty()) {
            const std::string valorTexto = truncar(
                texto, fonte, valor, tamanhoFonte, larguraValor - 16
            );
            texto.desenhar(
                imagem, fonte, valorTexto,
                x + larguraRotulo, y,
                tamanhoFonte, corTexto
            );
            registro[chave] = valorTexto;
        }

        y += alturaLinha;
        cv::line(
            imagem,
            cv::Point(x, y - 4),
            cv::Point(x + larguraUtil, y - 4),
            paraBgr(corLinha),
            1,
            cv::LINE_AA
        );
        ++indice;
    }

    cv::line(
        imagem,
        cv::Point(x + larguraRotulo - 6, yInicio - 4),
        cv::Point(x + larguraRotulo - 6, y - 4),
        paraBgr(corLinha),
        1,
        cv::LINE_AA
    );

    return y;
}

// ---------------------------------------------------------------------------
// Cabeçalho, rodapé e código de barras
// ---------------------------------------------------------------------------

void desenharCabecalho(
    Random& rng,
    TextRenderer& texto,
    cv::Mat& imagem,
    const std::string& titulo,
    const std::string& subtitulo,
    int x,
    int y,
    int larguraUtil,
    const std::string& fonte,
    int tamanhoTitulo,
    int tamanhoSub,
    const Rgb& corTitulo,
    const Rgb& corTexto,
    const Rgb& corLinha
) {
    const std::string alinhamento = escolha(rng, {"esquerda", "esquerda", "centro"});

    const int larguraTitulo = texto.largura(fonte, titulo, tamanhoTitulo);
    const int tx = alinhamento == "centro"
        ? x + std::max(0, (larguraUtil - larguraTitulo) / 2)
        : x;
    texto.desenhar(imagem, fonte, titulo, tx, y, tamanhoTitulo, corTitulo);
    y += static_cast<int>(tamanhoTitulo * 1.5);

    const int larguraSub = texto.largura(fonte, subtitulo, tamanhoSub);
    const int sx = alinhamento == "centro"
        ? x + std::max(0, (larguraUtil - larguraSub) / 2)
        : tx;
    texto.desenhar(imagem, fonte, subtitulo, sx, y, tamanhoSub, corTexto);
    y += static_cast<int>(tamanhoSub * 1.7);

    if (rng.chance(0.5)) {
        const int tamanhoSelo = escolhaInteiro(rng, {13, 15, 16});
        const std::string selo = escolha(rng, {
            "CONFIDENCIAL", "INTERNO", "PÚBLICO", "URGENTE",
            "RASCUNHO", "CONFERIDO",
        });
        const int larguraSelo = texto.largura("arialbd.ttf", selo, tamanhoSelo);
        texto.desenhar(
            imagem,
            "arialbd.ttf",
            selo,
            x + larguraUtil - larguraSelo,
            y - static_cast<int>(tamanhoTitulo * 1.5),
            tamanhoSelo,
            corTitulo
        );
    }

    if (rng.chance(0.4)) {
        cv::line(
            imagem,
            cv::Point(x, y),
            cv::Point(x + larguraUtil, y),
            paraBgr(corLinha),
            1,
            cv::LINE_AA
        );
        cv::line(
            imagem,
            cv::Point(x, y + 3),
            cv::Point(x + larguraUtil, y + 3),
            paraBgr(corLinha),
            1,
            cv::LINE_AA
        );
    }
}

void desenharCodigoBarras(
    Random& rng,
    TextRenderer& texto,
    cv::Mat& imagem,
    const std::string& codigo,
    int x,
    int y,
    const Rgb& corTexto,
    const VisualStyle& estilo
) {
    const int alturaBarras = rng.inteiro(24, 40);
    const int larguraTotal = rng.inteiro(120, 260);
    int cursor = x;

    while (cursor < x + larguraTotal) {
        const int larguraBarra = escolhaInteiro(rng, {1, 1, 2, 3});
        if (rng.chance(0.55)) {
            cv::rectangle(
                imagem,
                cv::Point(cursor, y),
                cv::Point(cursor + larguraBarra, y + alturaBarras),
                paraBgr(corTexto),
                cv::FILLED
            );
        }
        cursor += larguraBarra + escolhaInteiro(rng, {1, 2, 3});
    }

    texto.desenhar(imagem, estilo.fonte, codigo, x, y + alturaBarras + 2, 11, corTexto);
}

// ---------------------------------------------------------------------------
// Recorte e degradações
// ---------------------------------------------------------------------------

cv::Mat recortarParaConteudo(
    const cv::Mat& imagem,
    const Rgb& corFundo,
    int margemExtra
) {
    const int largura = imagem.cols;
    const int altura = imagem.rows;
    int ultimaLinha = -1;

    for (int y = altura - 1; y >= 0 && ultimaLinha < 0; --y) {
        const cv::Vec3b* linha = imagem.ptr<cv::Vec3b>(y);
        for (int x = 0; x < largura; ++x) {
            const int diferenca =
                std::abs(static_cast<int>(linha[x][0]) - corFundo.b) +
                std::abs(static_cast<int>(linha[x][1]) - corFundo.g) +
                std::abs(static_cast<int>(linha[x][2]) - corFundo.r);
            if (diferenca > 30) {
                ultimaLinha = y;
                break;
            }
        }
    }

    if (ultimaLinha < 0) {
        return imagem;
    }

    int novaAltura = std::min(altura, ultimaLinha + 1 + margemExtra);
    novaAltura = std::max(novaAltura, static_cast<int>(altura * 0.3));
    if (novaAltura >= altura) {
        return imagem;
    }

    return imagem(cv::Rect(0, 0, largura, novaAltura)).clone();
}

cv::Mat aplicarDegradacoes(Random& rng, cv::Mat imagem, const Rgb& corFundo) {
    if (rng.chance(0.35)) {
        const double sigma = rng.real(3.0, 12.0);
        std::normal_distribution<float> ruido(0.0f, static_cast<float>(sigma));
        cv::Mat flutuante;
        imagem.convertTo(flutuante, CV_32FC3);
        for (int y = 0; y < flutuante.rows; ++y) {
            cv::Vec3f* linha = flutuante.ptr<cv::Vec3f>(y);
            for (int x = 0; x < flutuante.cols; ++x) {
                for (int canal = 0; canal < 3; ++canal) {
                    linha[x][canal] = std::clamp(
                        linha[x][canal] + ruido(rng.motor()),
                        0.0f,
                        255.0f
                    );
                }
            }
        }
        flutuante.convertTo(imagem, CV_8UC3);
    }

    if (rng.chance(0.2)) {
        const double sigma = rng.real(0.3, 1.0);
        cv::GaussianBlur(imagem, imagem, cv::Size(3, 3), sigma);
    }

    if (rng.chance(0.3)) {
        const double angulo = rng.real(-2.5, 2.5);
        // cv::getRotationMatrix2D nao esta disponível nos cabeçalhos OpenCV
        // instalados neste ambiente (stub "opencv5" sem a funcao).
        // Reimplementa a mesma matriz afim 2x3 que o OpenCV gera (angulo
        // em graus, escala 1.0, em torno do centro da imagem).
        constexpr double PI = 3.14159265358979323846;
        const double rad = angulo * PI / 180.0;
        const double alpha = std::cos(rad);
        const double beta = std::sin(rad);
        const double cx = imagem.cols / 2.0;
        const double cy = imagem.rows / 2.0;
        cv::Mat rotacao = cv::Mat::zeros(2, 3, CV_64FC1);
        rotacao.at<double>(0, 0) = alpha;
        rotacao.at<double>(0, 1) = beta;
        rotacao.at<double>(0, 2) = (1.0 - alpha) * cx - beta * cy;
        rotacao.at<double>(1, 0) = -beta;
        rotacao.at<double>(1, 1) = alpha;
        rotacao.at<double>(1, 2) = beta * cx + (1.0 - alpha) * cy;
        cv::Mat girada;
        cv::warpAffine(
            imagem,
            girada,
            rotacao,
            imagem.size(),
            cv::INTER_CUBIC,
            cv::BORDER_CONSTANT,
            paraBgr(corFundo)
        );
        imagem = girada;
    }

    if (rng.chance(0.2)) {
        const float intensidade = static_cast<float>(rng.real(0.05, 0.22));
        const float centroX = imagem.cols / 2.0f;
        const float centroY = imagem.rows / 2.0f;
        cv::Mat flutuante;
        imagem.convertTo(flutuante, CV_32FC3);
        for (int y = 0; y < flutuante.rows; ++y) {
            cv::Vec3f* linha = flutuante.ptr<cv::Vec3f>(y);
            for (int x = 0; x < flutuante.cols; ++x) {
                const float dx = (static_cast<float>(x) - centroX) / centroX;
                const float dy = (static_cast<float>(y) - centroY) / centroY;
                const float distancia = std::sqrt(dx * dx + dy * dy);
                const float fator = 1.0f - intensidade * std::clamp(distancia, 0.0f, 1.5f);
                for (int canal = 0; canal < 3; ++canal) {
                    linha[x][canal] = std::clamp(
                        linha[x][canal] * fator,
                        0.0f,
                        255.0f
                    );
                }
            }
        }
        flutuante.convertTo(imagem, CV_8UC3);
    }

    if (rng.chance(0.25)) {
        cv::Mat cinza;
        cv::cvtColor(imagem, cinza, cv::COLOR_BGR2GRAY);
        const double media = cv::mean(cinza)[0];
        const double contraste = rng.real(0.8, 1.25);
        imagem.convertTo(imagem, -1, contraste, media * (1.0 - contraste));
        const double brilho = rng.real(0.9, 1.1);
        imagem.convertTo(imagem, -1, brilho, 0.0);
    }

    return imagem;
}

// ---------------------------------------------------------------------------
// Geração de documento
// ---------------------------------------------------------------------------

json gerarDocumento(
    Random& rng,
    long long indice,
    const fs::path& dirSaida,
    const std::string& tipoDoc,
    const std::optional<std::string>& forcarLayout,
    TextRenderer& texto
) {
    const DocProfile& perfil = perfilPorNome(tipoDoc);
    const VisualStyle& estilo = rng.escolher(ESTILOS_VISUAIS);
    const std::string layout = forcarLayout.has_value()
        ? *forcarLayout
        : rng.escolher(LAYOUTS);

    const FieldList campos = gerarCamposAleatorios(rng, tipoDoc);
    const std::string titulo = rng.escolher(perfil.titulos);
    const std::string subtitulo = rng.escolher(perfil.subtitulos);

    const int largura = escolhaInteiro(rng, {720, 800, 900, 1000, 1100});
    const int margem = rng.inteiro(28, 60);
    int altura = rng.inteiro(560, 900);
    altura = std::max(
        altura,
        margem * 2 + 260 + static_cast<int>(campos.size()) * 34
    );

    const int tamanhoTitulo = escolhaInteiro(rng, {24, 28, 32, 36});
    const int tamanhoSub = escolhaInteiro(rng, {15, 17, 19});
    const int tamanhoCampo = escolhaInteiro(rng, {15, 17, 18, 20});

    Rgb fundo = estilo.corFundo;
    if (estilo.fundoCinza) {
        fundo.r = std::max(0, fundo.r - rng.inteiro(0, 12));
        fundo.g = std::max(0, fundo.g - rng.inteiro(0, 12));
        fundo.b = std::max(0, fundo.b - rng.inteiro(0, 12));
    }

    cv::Mat imagem(altura, largura, CV_8UC3, paraBgr(fundo));

    const Rgb corTexto = estilo.corTexto;
    const Rgb corLinha = clarear(corTexto, 60);
    const Rgb corTitulo = rng.escolher(CORES_TITULO);
    const int alturaLinha = std::max(24, static_cast<int>(tamanhoCampo * 1.7));

    const int x = margem;
    const int larguraUtil = largura - margem * 2;
    int y = margem;

    desenharCabecalho(
        rng, texto, imagem, titulo, subtitulo,
        x, y, larguraUtil, estilo.fonte,
        tamanhoTitulo, tamanhoSub,
        corTitulo, corTexto, corLinha
    );

    y += escolhaInteiro(rng, {70, 84, 96});
    if (rng.chance(0.7)) {
        cv::line(
            imagem,
            cv::Point(x, y - 16),
            cv::Point(x + larguraUtil, y - 16),
            paraBgr(corLinha),
            2,
            cv::LINE_AA
        );
    }

    Registro registro;
    int yFim = y;

    if (layout == "lista") {
        yFim = renderizarLista(
            texto, imagem, campos, x, y, larguraUtil,
            estilo.fonte, tamanhoCampo, corTexto, corLinha,
            alturaLinha, registro
        );
    } else if (layout == "lista_alternada") {
        const Rgb fundoAlternado = escurecer(fundo, 12);
        yFim = renderizarListaAlternada(
            texto, imagem, campos, x, y, larguraUtil,
            estilo.fonte, tamanhoCampo, corTexto, corLinha,
            alturaLinha, fundoAlternado, registro
        );
    } else if (layout == "duas_colunas") {
        yFim = renderizarDuasColunas(
            texto, imagem, campos, x, y, larguraUtil,
            estilo.fonte, tamanhoCampo, corTexto, corLinha,
            alturaLinha, registro
        );
    } else if (layout == "tabela") {
        yFim = renderizarTabela(
            texto, imagem, campos, x, y, larguraUtil,
            estilo.fonte, tamanhoCampo, corTexto, corLinha,
            alturaLinha, registro
        );
    } else if (layout == "blocos") {
        yFim = renderizarBlocos(
            texto, imagem, campos, x, y, larguraUtil,
            estilo.fonte, tamanhoCampo, corTexto, corLinha,
            alturaLinha, registro
        );
    }

    for (const auto& [chave, valor] : campos) {
        if (registro.find(chave) == registro.end()) {
            registro[chave] = valor;
        }
    }

    if (rng.chance(0.5)) {
        int yRodape = std::min(altura - margem - 20, yFim + 20);
        const std::string rodape = escolha(rng, {
            "Documento processado eletronicamente.",
            "Página 1 de 1",
            "Emitido em sistema interno.",
            "Válido apenas com assinatura.",
        });
        texto.desenhar(imagem, estilo.fonte, rodape, x, yRodape, 12, corTexto);
        yRodape += 18;

        if (rng.chance(0.6)) {
            const int numero = rng.inteiro(10000, 99999);
            const std::string codigo = "REF-" + std::to_string(numero);
            desenharCodigoBarras(
                rng, texto, imagem, codigo,
                x, yRodape, corTexto, estilo
            );
        }
    }

    imagem = recortarParaConteudo(imagem, fundo, rng.inteiro(16, 40));
    imagem = aplicarDegradacoes(rng, imagem, estilo.corFundo);

    std::ostringstream nomeArquivo;
    nomeArquivo << "doc_" << std::setw(6) << std::setfill('0') << indice << ".png";
    const std::string nomeImagem = nomeArquivo.str();

    const fs::path dirImagens = dirSaida / "images";
    const fs::path dirAnotacoes = dirSaida / "annotations";
    fs::create_directories(dirImagens);
    fs::create_directories(dirAnotacoes);

    const fs::path caminhoImagem = dirImagens / nomeImagem;
    if (!cv::imwrite(caminhoImagem.string(), imagem)) {
        throw std::runtime_error(
            "Não foi possível salvar a imagem: " + caminhoImagem.string()
        );
    }

    json anotacao;
    anotacao["image"] = nomeImagem;
    json camposJson = json::object();
    for (const auto& [chave, valor] : registro) {
        camposJson[chave] = valor;
    }
    anotacao["fields"] = camposJson;

    const fs::path caminhoAnotacao =
        dirAnotacoes / (fs::path(nomeImagem).stem().string() + ".json");
    std::ofstream arquivo(caminhoAnotacao.string(), std::ios::binary);
    if (!arquivo) {
        throw std::runtime_error(
            "Não foi possível criar a anotação: " + caminhoAnotacao.string()
        );
    }
    arquivo << anotacao.dump(2);
    if (!arquivo) {
        throw std::runtime_error(
            "Falha ao gravar a anotação: " + caminhoAnotacao.string()
        );
    }

    return anotacao;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Options {
    fs::path output = "./training_data";
    long long num = 1000000;
    long long seed = 42;
    std::string types = "all";
    std::string layout = "all";
    bool balancedLayouts = false;
    bool balancedTypes = false;
    long long startIndex = 0;
    bool limpar = false;
};

class ArgumentError : public std::runtime_error {
public:
    explicit ArgumentError(const std::string& mensagem)
        : std::runtime_error(mensagem) {}
};

void imprimirUso(std::ostream& saida) {
    saida << "Uso: gerar_itens_treino [opções]\n\n";
    saida << "Opções:\n";
    saida << "  -o, --output DIR        Diretório de saída (padrão: ./training_data)\n";
    saida << "  -n, --num N             Documentos a gerar (padrão: 1000000)\n";
    saida << "      --seed N            Semente do gerador (padrão: 42)\n";
    saida << "      --types LISTA       Tipos separados por vírgula ou 'all' (padrão: all)\n";
    saida << "      --layout NOME       Layout fixo ou 'all' (padrão: all)\n";
    saida << "      --balanced-layouts  Distribui layouts em round-robin\n";
    saida << "      --balanced-types    Distribui tipos em round-robin\n";
    saida << "      --start-index N     Índice inicial dos arquivos (padrão: 0)\n";
    saida << "      --limpar            Remove imagens/anotações anteriores\n";
    saida << "  -h, --help              Mostra esta ajuda\n";
    saida << "\nTipos: " << juntar(nomesDosTipos(), ", ") << '\n';
    saida << "Layouts: " << juntar(LAYOUTS, ", ") << '\n';
}

long long parseLong(const std::string& texto, const std::string& opcao) {
    std::size_t consumido = 0;
    long long valor = 0;
    try {
        valor = std::stoll(texto, &consumido);
    } catch (const std::exception&) {
        throw ArgumentError("Valor inválido para " + opcao + ": " + texto);
    }
    if (consumido != texto.size()) {
        throw ArgumentError("Valor inválido para " + opcao + ": " + texto);
    }
    return valor;
}

Options parseArgumentos(int argc, char** argv) {
    Options opcoes;

    for (int i = 1; i < argc; ++i) {
        std::string argumento = argv[i];
        if (argumento == "--help" || argumento == "-h") {
            imprimirUso(std::cout);
            std::exit(0);
        }

        std::string valorInline;
        bool temValorInline = false;
        const auto separador = argumento.find('=');
        if (separador != std::string::npos) {
            valorInline = argumento.substr(separador + 1);
            argumento = argumento.substr(0, separador);
            temValorInline = true;
        }

        auto obterValor = [&]() -> std::string {
            if (temValorInline) {
                return valorInline;
            }
            if (i + 1 >= argc) {
                throw ArgumentError("Faltou valor para " + argumento);
            }
            return std::string(argv[++i]);
        };

        if (argumento == "--output" || argumento == "-o") {
            opcoes.output = obterValor();
        } else if (argumento == "--num" || argumento == "-n") {
            opcoes.num = parseLong(obterValor(), argumento);
        } else if (argumento == "--seed") {
            opcoes.seed = parseLong(obterValor(), argumento);
        } else if (argumento == "--types") {
            opcoes.types = obterValor();
        } else if (argumento == "--layout") {
            opcoes.layout = obterValor();
        } else if (argumento == "--balanced-layouts") {
            opcoes.balancedLayouts = true;
        } else if (argumento == "--balanced-types") {
            opcoes.balancedTypes = true;
        } else if (argumento == "--start-index") {
            opcoes.startIndex = parseLong(obterValor(), argumento);
        } else if (argumento == "--limpar") {
            opcoes.limpar = true;
        } else {
            throw ArgumentError("Opção desconhecida: " + argumento);
        }
    }

    return opcoes;
}

// ---------------------------------------------------------------------------
// Execução
// ---------------------------------------------------------------------------

struct ContagemImagens {
    long long quantidade = 0;
    long long maiorIndice = -1;
};

ContagemImagens listarImagensExistentes(const fs::path& diretorio) {
    ContagemImagens contagem;
    if (!fs::is_directory(diretorio)) {
        return contagem;
    }

    try {
        for (const auto& entrada : fs::directory_iterator(diretorio)) {
            if (!entrada.is_regular_file()) {
                continue;
            }
            const fs::path caminho = entrada.path();
            const std::string nome = caminho.filename().string();
            if (nome.rfind("doc_", 0) != 0 || caminho.extension() != ".png") {
                continue;
            }
            ++contagem.quantidade;
            try {
                const long long indice = std::stoll(
                    caminho.stem().string().substr(4)
                );
                contagem.maiorIndice = std::max(contagem.maiorIndice, indice);
            } catch (const std::exception&) {
            }
        }
    } catch (const fs::filesystem_error&) {
    }

    return contagem;
}

void limparArquivos(const fs::path& diretorio) {
    if (!fs::is_directory(diretorio)) {
        return;
    }
    for (const auto& entrada : fs::directory_iterator(diretorio)) {
        if (entrada.is_regular_file()) {
            fs::remove(entrada.path());
        }
    }
}

template <typename T>
std::vector<std::pair<std::string, T>> ordenarPorQuantidade(
    const std::map<std::string, T>& contagem
) {
    std::vector<std::pair<std::string, T>> itens(
        contagem.begin(), contagem.end()
    );
    std::stable_sort(
        itens.begin(), itens.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        }
    );
    return itens;
}

int executar(const Options& opcoes) {
    std::vector<std::string> tipos;
    if (opcoes.types == "all") {
        tipos = nomesDosTipos();
    } else {
        for (std::string tipo : dividir(opcoes.types, ',')) {
            tipo = aparar(tipo);
            if (!tipo.empty()) {
                tipos.push_back(tipo);
            }
        }

        std::vector<std::string> invalidos;
        for (const auto& tipo : tipos) {
            if (!tipoExiste(tipo)) {
                invalidos.push_back(tipo);
            }
        }
        if (!invalidos.empty()) {
            std::cerr << "Tipos inválidos: " << juntar(invalidos, ", ") << '\n';
            std::cerr << "Tipos disponíveis: " << juntar(nomesDosTipos(), ", ") << '\n';
            return 1;
        }
        if (tipos.empty()) {
            std::cerr << "Nenhum tipo informado em --types.\n";
            return 1;
        }
    }

    if (opcoes.layout != "all" && !layoutExiste(opcoes.layout)) {
        std::cerr << "Layout inválido: " << opcoes.layout << '\n';
        std::cerr << "Layouts disponíveis: " << juntar(LAYOUTS, ", ") << '\n';
        return 1;
    }

    const fs::path dirSaida = opcoes.output;
    fs::create_directories(dirSaida / "images");
    fs::create_directories(dirSaida / "annotations");

    if (opcoes.limpar) {
        limparArquivos(dirSaida / "images");
        limparArquivos(dirSaida / "annotations");
        std::cout << "Conteúdo anterior removido de " << dirSaida.string() << '\n';
    }

    const ContagemImagens existentes = listarImagensExistentes(dirSaida / "images");
    long long proximoIndice = opcoes.startIndex;
    if (existentes.quantidade > 0 && existentes.maiorIndice >= 0) {
        proximoIndice = std::max(proximoIndice, existentes.maiorIndice + 1);
    }

    const long long restante = opcoes.num - existentes.quantidade;

    std::cout << "============================================================\n";
    std::cout << "Gerador de documentos sintéticos genéricos\n";
    std::cout << "============================================================\n";
    std::cout << "Saída: " << dirSaida.string() << '\n';
    std::cout << "Quantidade solicitada: " << opcoes.num << '\n';
    std::cout << "Imagens já existentes: " << existentes.quantidade << '\n';

    if (restante <= 0) {
        std::cout << "Nenhum documento novo necessário. O alvo já foi atingido.\n";
        std::cout << "============================================================\n";
        return 0;
    }

    std::cout << "Documentos a gerar: " << restante << '\n';
    std::cout << "Índice inicial: " << proximoIndice << '\n';
    std::cout << "Tipos: " << juntar(tipos, ", ") << '\n';
    std::cout << "Layouts: " << opcoes.layout << '\n';
    if (opcoes.balancedLayouts) {
        std::cout << "Layouts balanceados: Sim (round-robin)\n";
    }
    if (opcoes.balancedTypes) {
        std::cout << "Tipos balanceados: Sim (round-robin)\n";
    }
    std::cout << "Semente: " << opcoes.seed << '\n';
    std::cout << "============================================================\n";

    Random rng(static_cast<std::uint32_t>(opcoes.seed));
    TextRenderer texto;

    const std::vector<std::string>* cicloTipos =
        opcoes.balancedTypes ? &tipos : nullptr;
    const std::vector<std::string>* cicloLayouts =
        (opcoes.balancedLayouts && opcoes.layout == "all") ? &LAYOUTS : nullptr;

    std::set<std::string> rotulosVistos;
    std::map<std::string, long long> contagemTipos;
    std::map<std::string, long long> contagemLayouts;

    const auto inicio = std::chrono::steady_clock::now();
    auto ultimoProgresso = inicio;
    constexpr double intervaloProgresso = 0.2;

    for (long long i = 0; i < restante; ++i) {
        std::string tipo;
        if (cicloTipos != nullptr) {
            tipo = (*cicloTipos)[static_cast<std::size_t>(i) % cicloTipos->size()];
        } else {
            const int indiceTipo = rng.inteiro(0, static_cast<int>(tipos.size()) - 1);
            tipo = tipos[static_cast<std::size_t>(indiceTipo)];
        }

        std::optional<std::string> layout;
        if (cicloLayouts != nullptr) {
            layout = (*cicloLayouts)[static_cast<std::size_t>(i) % cicloLayouts->size()];
        } else if (opcoes.layout != "all") {
            layout = opcoes.layout;
        }

        const json anotacao = gerarDocumento(
            rng,
            proximoIndice + i,
            dirSaida,
            tipo,
            layout,
            texto
        );

        for (const auto& item : anotacao["fields"].items()) {
            rotulosVistos.insert(item.key());
        }
        ++contagemTipos[tipo];
        if (layout.has_value()) {
            ++contagemLayouts[*layout];
        } else {
            ++contagemLayouts["<aleatorio>"];
        }

        const auto agora = std::chrono::steady_clock::now();
        const double decorrido =
            std::chrono::duration<double>(agora - inicio).count();
        const double desdeUltimo =
            std::chrono::duration<double>(agora - ultimoProgresso).count();

        if (i + 1 == restante || desdeUltimo >= intervaloProgresso) {
            const double taxa = static_cast<double>(i + 1) / std::max(decorrido, 1e-9);
            const double segundosRestantes =
                static_cast<double>(restante - (i + 1)) / std::max(taxa, 1e-9);
            const double percentual =
                100.0 * static_cast<double>(i + 1) / static_cast<double>(restante);

            std::ostringstream linha;
            linha.imbue(std::locale::classic());
            linha << "\rGerando documentos: " << (i + 1) << "/" << restante
                  << " (" << std::fixed << std::setprecision(1) << percentual
                  << "%) | " << std::setprecision(2) << taxa << " doc/s"
                  << " | decorrido " << formatarDuracao(decorrido)
                  << " | ETA " << formatarDuracao(segundosRestantes);
            std::cout << linha.str() << std::flush;
            ultimoProgresso = agora;
        }
    }

    std::cout << "\n\n";
    std::cout << "============================================================\n";
    std::cout << "Geração concluída!\n";
    std::cout << "Imagens: " << (dirSaida / "images").string() << '\n';
    std::cout << "Anotações: " << (dirSaida / "annotations").string() << '\n';
    std::cout << "Documentos: " << opcoes.num << '\n';
    std::cout << "Gerados agora: " << restante << '\n';
    std::cout << "Rótulos distintos: " << rotulosVistos.size() << '\n';

    std::cout << "Distribuição por tipo:\n";
    for (const auto& [tipo, quantidade] : ordenarPorQuantidade(contagemTipos)) {
        std::cout << "     - " << tipo << ": " << quantidade << '\n';
    }

    std::cout << "Distribuição por layout:\n";
    for (const auto& [layout, quantidade] : ordenarPorQuantidade(contagemLayouts)) {
        std::cout << "     - " << layout << ": " << quantidade << '\n';
    }

    std::cout << "============================================================\n\n";
    std::cout << "Próximos passos:\n";
    std::cout << "1. Verifique o dataset: py training/prepare_dataset.py --verify "
              << (dirSaida / "images").string() << " "
              << (dirSaida / "annotations").string() << '\n';
    std::cout << "2. Treine o modelo:    py main.py train --images "
              << (dirSaida / "images").string()
              << " --annotations " << (dirSaida / "annotations").string() << '\n';

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try {
        const Options opcoes = parseArgumentos(argc, argv);
        return executar(opcoes);
    } catch (const ArgumentError& erro) {
        std::cerr << "Erro: " << erro.what() << "\n\n";
        imprimirUso(std::cerr);
        return 2;
    } catch (const std::exception& erro) {
        std::cerr << "Erro: " << erro.what() << '\n';
        return 1;
    }
}
