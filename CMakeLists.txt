cmake_minimum_required(VERSION 3.20)

project(analizeImage_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

option(BUILD_EXAMPLES "Compila os exemplos da pasta examples" ON)

# No MSYS2/MinGW, o prefixo do toolchain (ex.: D:/msys64/mingw64)
# não entra no CMAKE_PREFIX_PATH automaticamente com o CMake nativo.
if(MINGW)
    get_filename_component(_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_mingw_prefix "${_mingw_bin}" DIRECTORY)
    list(APPEND CMAKE_PREFIX_PATH "${_mingw_prefix}")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs freetype)
message(STATUS "OpenCV ${OpenCV_VERSION} encontrado em ${OpenCV_DIR}")

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
    if(NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR
            "nlohmann/json.hpp não encontrado. "
            "No MSYS2: pacman -S mingw-w64-x86_64-nlohmann-json"
        )
    endif()
endif()

add_executable(gerar_itens_treino gerar_itens_treino.cpp)
target_include_directories(gerar_itens_treino PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gerar_itens_treino PRIVATE ${OpenCV_LIBS})

if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(gerar_itens_treino PRIVATE nlohmann_json::nlohmann_json)
else()
    target_include_directories(gerar_itens_treino PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

if(MSVC)
    target_compile_options(gerar_itens_treino PRIVATE /utf-8)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
