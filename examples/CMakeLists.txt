add_executable(create_test_doc create_test_doc.cpp)
target_include_directories(create_test_doc PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(create_test_doc PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(create_test_doc PRIVATE /utf-8)
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(TESSERACT IMPORTED_TARGET tesseract)
    pkg_check_modules(LEPTONICA IMPORTED_TARGET lept)
endif()

if(PkgConfig_FOUND AND TESSERACT_FOUND AND LEPTONICA_FOUND)
    add_executable(example_usage example_usage.cpp)
    target_link_libraries(example_usage PRIVATE
        PkgConfig::TESSERACT
        PkgConfig::LEPTONICA
    )

    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(example_usage PRIVATE nlohmann_json::nlohmann_json)
    elseif(NLOHMANN_JSON_INCLUDE_DIR)
        target_include_directories(example_usage PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
    endif()

    if(MSVC)
        target_compile_options(example_usage PRIVATE /utf-8)
    endif()
else()
    message(WARNING
        "example_usage não será compilado: pkg-config, Tesseract ou Leptonica "
        "não encontrados. Instale mingw-w64-x86_64-pkgconf, "
        "mingw-w64-x86_64-tesseract-ocr e mingw-w64-x86_64-leptonica."
    )
endif()

add_custom_target(exemplos DEPENDS create_test_doc)
if(TARGET example_usage)
    add_dependencies(exemplos example_usage)
endif()
