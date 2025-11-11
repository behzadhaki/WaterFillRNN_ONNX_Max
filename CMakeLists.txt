cmake_minimum_required(VERSION 3.16)

project(WaterFillRNN
        VERSION 1.0.0
        LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# Build Settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Windows-specific settings
if(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_definitions(-D_WIN32_WINNT=0x0601)  # Windows 7 minimum
    add_definitions(-DNOMINMAX)  # Prevent min/max macro conflicts
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    # Prevent conflict between MSVC's round() and Jitter's round()
endif()
if (MSVC)
    add_definitions(-D_CRT_NONSTDC_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DUSE_JITTER_ROUND)
endif()
# ------------------------------------------------------------------------------
# Platform Detection and Configuration
# ------------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PLATFORM_WINDOWS TRUE)
    set(PLATFORM_MACOS FALSE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(PLATFORM_WINDOWS FALSE)
    set(PLATFORM_MACOS TRUE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# ------------------------------------------------------------------------------
# Codesigning Configuration (macOS only)
# ------------------------------------------------------------------------------
if(PLATFORM_MACOS)
    set(CODESIGN_IDENTITY "" CACHE STRING "Code signing identity (empty for ad-hoc signing)")
    set(ENABLE_CODESIGNING ON CACHE BOOL "Enable automatic codesigning")

    if(ENABLE_CODESIGNING)
        if(CODESIGN_IDENTITY STREQUAL "")
            message(STATUS "Codesigning: Ad-hoc signing enabled")
            set(CODESIGN_FLAGS --force --deep -s -)
        else()
            message(STATUS "Codesigning: Using identity '${CODESIGN_IDENTITY}'")
            set(CODESIGN_FLAGS --force --options runtime --timestamp --deep -s "${CODESIGN_IDENTITY}")
        endif()
    else()
        message(STATUS "Codesigning: Disabled")
    endif()
endif()

# ------------------------------------------------------------------------------
# Fetch CPM
# ------------------------------------------------------------------------------
include(cmake/CPM.cmake)

# ------------------------------------------------------------------------------
# Fetch Min API
# ------------------------------------------------------------------------------
CPMAddPackage(
        NAME MinAPI
        GITHUB_REPOSITORY Cycling74/min-api
        GIT_TAG main
)
if (NOT MinAPI_ADDED)
    message(FATAL_ERROR "Failed to fetch Min API.")
endif()
set(MIN_API_ROOT ${MinAPI_SOURCE_DIR})
message(STATUS "Using Min API from: ${MIN_API_ROOT}")

# ------------------------------------------------------------------------------
# Ensure Max SDK
# ------------------------------------------------------------------------------
if(PLATFORM_WINDOWS)
    set(MAX_SDK_ROOT "${CMAKE_SOURCE_DIR}/lib/max-sdk")
    if(NOT EXISTS "${MAX_SDK_ROOT}/source/max-sdk-base")
        message(STATUS "Max SDK not found; running clone_maxsdk.bat ...")
        execute_process(
                COMMAND cmd /c "${CMAKE_SOURCE_DIR}/clone_maxsdk.bat"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                RESULT_VARIABLE _clone_res
        )
        if(NOT _clone_res EQUAL 0)
            message(FATAL_ERROR "Failed to clone Max SDK via clone_maxsdk.bat")
        endif()
    endif()
else()
    set(MAX_SDK_ROOT "${CMAKE_SOURCE_DIR}/lib/max-sdk")
    if(NOT EXISTS "${MAX_SDK_ROOT}/source/max-sdk-base")
        message(STATUS "Max SDK not found; running clone_maxsdk.sh ...")
        execute_process(
                COMMAND bash "${CMAKE_SOURCE_DIR}/clone_maxsdk.sh"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                RESULT_VARIABLE _clone_res
        )
        if(NOT _clone_res EQUAL 0)
            message(FATAL_ERROR "Failed to clone Max SDK via clone_maxsdk.sh")
        endif()
    endif()
endif()
message(STATUS "Using Max SDK from: ${MAX_SDK_ROOT}")

# ------------------------------------------------------------------------------
# ONNX Runtime Configuration
# ------------------------------------------------------------------------------
if(PLATFORM_WINDOWS)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ONNX_ARCHIVE_NAME "onnxruntime-win-x64-static_lib-1.20.1.tar.bz2")
        set(ONNX_SUBDIR_NAME  "onnxruntime-win-x64-static_lib-1.20.1")
    else()
        message(FATAL_ERROR "32-bit Windows builds are not supported")
    endif()
elseif(PLATFORM_MACOS)
    if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(ONNX_ARCHIVE_NAME "onnxruntime-osx-x86_64-static_lib-1.22.0.zip")
        set(ONNX_SUBDIR_NAME  "onnxruntime-osx-x86_64-static_lib-1.22.0")
    else()
        set(ONNX_ARCHIVE_NAME "onnxruntime-osx-arm64-static_lib-1.20.0.zip")
        set(ONNX_SUBDIR_NAME  "onnxruntime-osx-arm64-static_lib-1.20.0")
    endif()
endif()

set(ONNX_ARCHIVE_FILE "${CMAKE_SOURCE_DIR}/${ONNX_ARCHIVE_NAME}")
if(NOT EXISTS "${ONNX_ARCHIVE_FILE}")
    message(FATAL_ERROR "ONNX archive not found: ${ONNX_ARCHIVE_FILE}")
endif()

set(ONNX_UNPACK_DIR "${CMAKE_BINARY_DIR}/external/onnxruntime_prebuilt")
set(ONNX_ROOT_DIR   "${ONNX_UNPACK_DIR}/${ONNX_SUBDIR_NAME}")

# Check if already unpacked
if(PLATFORM_WINDOWS)
    set(ONNX_HEADER_CHECK "${ONNX_ROOT_DIR}/include/onnxruntime_c_api.h")
else()
    set(ONNX_HEADER_CHECK "${ONNX_ROOT_DIR}/include/onnxruntime_c_api.h")
endif()

if(NOT EXISTS "${ONNX_HEADER_CHECK}")
    message(STATUS "Unpacking ONNX Runtime from: ${ONNX_ARCHIVE_FILE}")
    file(MAKE_DIRECTORY "${ONNX_UNPACK_DIR}")

    if(PLATFORM_WINDOWS)
        # Try multiple extraction methods for Windows tar.bz2
        set(EXTRACTION_SUCCESS FALSE)

        # Method 1: Try 7-Zip if available
        find_program(SEVEN_ZIP_EXECUTABLE 7z
                PATHS "C:/Program Files/7-Zip" "C:/Program Files (x86)/7-Zip")

        if(SEVEN_ZIP_EXECUTABLE AND NOT EXTRACTION_SUCCESS)
            message(STATUS "Found 7-Zip: ${SEVEN_ZIP_EXECUTABLE}")
            message(STATUS "Attempting extraction with 7-Zip...")

            # Step 1: Extract the .tar from the .tar.bz2
            execute_process(
                    COMMAND ${SEVEN_ZIP_EXECUTABLE} x "${ONNX_ARCHIVE_FILE}" -o"${ONNX_UNPACK_DIR}" -y
                    RESULT_VARIABLE SEVEN_ZIP_RES1
                    OUTPUT_QUIET
                    ERROR_QUIET
            )

            # Step 2: If step 1 succeeded, extract the .tar contents
            if(SEVEN_ZIP_RES1 EQUAL 0)
                # Find the .tar file that was just created
                file(GLOB ONNX_TAR_FILE "${ONNX_UNPACK_DIR}/*.tar")
                if(ONNX_TAR_FILE)
                    execute_process(
                            COMMAND ${SEVEN_ZIP_EXECUTABLE} x "${ONNX_TAR_FILE}" -o"${ONNX_UNPACK_DIR}" -y
                            RESULT_VARIABLE SEVEN_ZIP_RES2
                            OUTPUT_QUIET
                            ERROR_QUIET
                    )
                    if(SEVEN_ZIP_RES2 EQUAL 0)
                        set(EXTRACTION_SUCCESS TRUE)
                        message(STATUS "Successfully extracted with 7-Zip")
                        # Optionally remove the intermediate .tar
                        file(REMOVE ${ONNX_TAR_FILE})
                    endif()
                endif()
            endif()
        endif()


        # Method 2: Try WinRAR if available
        find_program(WINRAR_EXECUTABLE winrar PATHS "C:/Program Files/WinRAR" "C:/Program Files (x86)/WinRAR")
        if(WINRAR_EXECUTABLE AND NOT EXTRACTION_SUCCESS)
            message(STATUS "Attempting extraction with WinRAR...")
            execute_process(
                    COMMAND ${WINRAR_EXECUTABLE} x -y "${ONNX_ARCHIVE_FILE}" "${ONNX_UNPACK_DIR}/"
                    RESULT_VARIABLE WINRAR_RES
                    OUTPUT_QUIET
                    ERROR_QUIET
            )
            if(WINRAR_RES EQUAL 0)
                set(EXTRACTION_SUCCESS TRUE)
                message(STATUS "Successfully extracted with WinRAR")
            endif()
        endif()

        # Method 3: Try tar with explicit bzip2 if available
        find_program(TAR_EXECUTABLE tar)
        find_program(BZIP2_EXECUTABLE bzip2)
        if(TAR_EXECUTABLE AND BZIP2_EXECUTABLE AND NOT EXTRACTION_SUCCESS)
            message(STATUS "Attempting extraction with tar and bzip2...")
            execute_process(
                    COMMAND ${TAR_EXECUTABLE} -xjf "${ONNX_ARCHIVE_FILE}" -C "${ONNX_UNPACK_DIR}"
                    RESULT_VARIABLE TAR_RES
                    OUTPUT_QUIET
                    ERROR_QUIET
            )
            if(TAR_RES EQUAL 0)
                set(EXTRACTION_SUCCESS TRUE)
                message(STATUS "Successfully extracted with tar")
            endif()
        endif()

        # Method 4: Try PowerShell with .NET compression (fallback)
        if(NOT EXTRACTION_SUCCESS)
            message(STATUS "Attempting extraction with PowerShell...")
            set(PS_SCRIPT "
                try {
                    Add-Type -AssemblyName System.IO.Compression.FileSystem
                    $archive = '${ONNX_ARCHIVE_FILE}'
                    $destination = '${ONNX_UNPACK_DIR}'

                    # Note: PowerShell can't handle .tar.bz2 directly
                    # This is a fallback that will likely fail but provides a clear error
                    Write-Host 'PowerShell cannot extract .tar.bz2 files directly'
                    exit 1
                } catch {
                    Write-Host 'PowerShell extraction failed:' $_.Exception.Message
                    exit 1
                }
            ")
            execute_process(
                    COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command "${PS_SCRIPT}"
                    RESULT_VARIABLE PS_RES
                    OUTPUT_QUIET
                    ERROR_QUIET
            )
            if(PS_RES EQUAL 0)
                set(EXTRACTION_SUCCESS TRUE)
                message(STATUS "Successfully extracted with PowerShell")
            endif()
        endif()

        if(NOT EXTRACTION_SUCCESS)
            message(FATAL_ERROR
                    "Failed to extract ${ONNX_ARCHIVE_FILE}. Please install one of the following:\n"
                    "  - 7-Zip (https://www.7-zip.org/)\n"
                    "  - WinRAR (https://www.win-rar.com/)\n"
                    "  - Git for Windows (which includes bzip2)\n"
                    "Or manually extract the archive to: ${ONNX_UNPACK_DIR}")
        endif()
    else()
        # macOS zip extraction
        execute_process(
                COMMAND unzip -o -q "${ONNX_ARCHIVE_FILE}" -d "${ONNX_UNPACK_DIR}"
                RESULT_VARIABLE UNZIP_RES
        )
        if(NOT UNZIP_RES EQUAL 0)
            message(FATAL_ERROR "Failed to unzip ${ONNX_ARCHIVE_FILE}")
        endif()
    endif()
endif()
message(STATUS "ONNX unpacked to: ${ONNX_ROOT_DIR}")

# Find ONNX Runtime library
if(PLATFORM_WINDOWS)
    # Windows uses static .lib files
    if(EXISTS "${ONNX_ROOT_DIR}/lib/onnxruntime.lib")
        set(ONNX_LIB "${ONNX_ROOT_DIR}/lib/onnxruntime.lib")
        set(ONNX_IS_STATIC TRUE)
    else()
        message(FATAL_ERROR "Could not find onnxruntime.lib in ${ONNX_ROOT_DIR}/lib")
    endif()
else()
    # macOS - prefer static, fall back to dylib
    if(EXISTS "${ONNX_ROOT_DIR}/lib/libonnxruntime.a")
        set(ONNX_LIB "${ONNX_ROOT_DIR}/lib/libonnxruntime.a")
        set(ONNX_IS_STATIC TRUE)
    elseif(EXISTS "${ONNX_ROOT_DIR}/lib/libonnxruntime.dylib")
        set(ONNX_LIB "${ONNX_ROOT_DIR}/lib/libonnxruntime.dylib")
        set(ONNX_IS_STATIC FALSE)
    else()
        message(FATAL_ERROR "Could not find libonnxruntime in ${ONNX_ROOT_DIR}/lib")
    endif()
endif()

message(STATUS "ONNX headers: ${ONNX_ROOT_DIR}/include")
message(STATUS "ONNX lib:     ${ONNX_LIB}")

# ------------------------------------------------------------------------------
# Common Includes
# ------------------------------------------------------------------------------
set(COMMON_INCLUDES
        ${MIN_API_ROOT}/include
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes
        ${ONNX_ROOT_DIR}/include
        ${CMAKE_SOURCE_DIR}/include           # onnx_helpers.h
)

# ------------------------------------------------------------------------------
# Shared Source Directory
# ------------------------------------------------------------------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# ------------------------------------------------------------------------------
# Platform-specific frameworks and libraries
# ------------------------------------------------------------------------------
if(PLATFORM_WINDOWS)
    # Windows system libraries needed by ONNX Runtime
    set(PLATFORM_LIBRARIES
            kernel32
            user32
            gdi32
            winspool
            shell32
            ole32
            oleaut32
            uuid
            comdlg32
            advapi32
            winmm
            ws2_32
            crypt32
            bcrypt
    )
else()
    # Apple frameworks (ORT static uses CF/NS)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    find_library(FOUNDATION_FRAMEWORK     Foundation)
    find_library(CORESERVICES_FRAMEWORK   CoreServices)
    find_library(SECURITY_FRAMEWORK       Security)
    find_library(SYSCONFIG_FRAMEWORK      SystemConfiguration)
    find_library(OBJC_LIB                 objc)

    set(PLATFORM_LIBRARIES
            ${COREFOUNDATION_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${CORESERVICES_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSCONFIG_FRAMEWORK}
            ${OBJC_LIB}
    )
endif()

# ------------------------------------------------------------------------------
# Max user Packages destination
# ------------------------------------------------------------------------------
if(PLATFORM_WINDOWS)
    set(MAX_VERSION "Max 9" CACHE STRING "Max user folder name, e.g. 'Max 8' or 'Max 9'")
    set(MAX_PACKAGE_NAME "WaterFill" CACHE STRING "Package folder name under Documents/${MAX_VERSION}/Packages")
    set(MAX_USER_PACKAGES_DIR "$ENV{USERPROFILE}/Documents/${MAX_VERSION}/Packages" CACHE PATH "Max Packages directory")
    set(EXTERNAL_EXTENSION "mxe64")
else()
    set(MAX_VERSION "Max 9" CACHE STRING "Max user folder name, e.g. 'Max 8' or 'Max 9'")
    set(MAX_PACKAGE_NAME "WaterFill" CACHE STRING "Package folder name under ~/Documents/${MAX_VERSION}/Packages")
    set(MAX_USER_PACKAGES_DIR "$ENV{HOME}/Documents/${MAX_VERSION}/Packages" CACHE PATH "Max Packages directory")
    set(EXTERNAL_EXTENSION "mxo")
endif()

set(MAX_PACKAGE_DIR "${MAX_USER_PACKAGES_DIR}/${MAX_PACKAGE_NAME}")
set(MAX_EXTERNALS_DIR "${MAX_PACKAGE_DIR}/externals")
set(MAX_RESOURCES_DIR "${MAX_PACKAGE_DIR}/resources")

message(STATUS "Max externals will be copied to: ${MAX_EXTERNALS_DIR}")
message(STATUS "Max resources will be copied to: ${MAX_RESOURCES_DIR}")

# ------------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------------
set(MODEL_VARIANTS
        "onnx/encodec_decode.onnx"
        "onnx/rnn_step.onnx"
)

set(ALL_MODEL_VARIANTS ${MODEL_VARIANTS})

# ------------------------------------------------------------------------------
# Helper function for Windows/macOS deployment
# ------------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Helper function for Windows/macOS deployment
# --------------------------------------------------------------------------
function(setup_external_deployment TARGET_NAME)
    if(PLATFORM_WINDOWS)
        set(BUNDLE_PATH "${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}")

        # Copy any required DLLs if using shared libraries (not needed for static)
        if(NOT ONNX_IS_STATIC AND EXISTS "${ONNX_ROOT_DIR}/lib/onnxruntime.dll")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNX_ROOT_DIR}/lib/onnxruntime.dll"
                    "${CMAKE_BINARY_DIR}/"
                    COMMENT "Copied onnxruntime.dll for ${TARGET_NAME}"
            )
        endif()

        # On Windows, just copy the .mxe64 file
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_EXTERNALS_DIR}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${BUNDLE_PATH}"
                "${MAX_EXTERNALS_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}"
                COMMENT "Deployed ${TARGET_NAME}.${EXTERNAL_EXTENSION} to ${MAX_EXTERNALS_DIR}"
        )

    else()
        set(BUNDLE_PATH "${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}")

        # Copy dylib into bundle if using shared ORT
        if(NOT ONNX_IS_STATIC)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNX_LIB}"
                    "${BUNDLE_PATH}/Contents/MacOS/"
                    COMMENT "Copied libonnxruntime.dylib to ${TARGET_NAME}.${EXTERNAL_EXTENSION}/Contents/MacOS"
            )
        endif()

        # Remove quarantine attributes and codesign (macOS only)
        if(ENABLE_CODESIGNING)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND xattr -r -d com.apple.quarantine "${BUNDLE_PATH}" || true
                    COMMAND codesign ${CODESIGN_FLAGS} "${BUNDLE_PATH}"
                    COMMENT "Codesigning ${TARGET_NAME}.${EXTERNAL_EXTENSION}"
                    VERBATIM
            )
        endif()

        # On macOS, copy the bundle directory
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_EXTERNALS_DIR}"
                COMMAND ${CMAKE_COMMAND} -E remove_directory "${MAX_EXTERNALS_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${BUNDLE_PATH}"
                "${MAX_EXTERNALS_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}"
                COMMENT "Deployed ${TARGET_NAME}.${EXTERNAL_EXTENSION} to ${MAX_EXTERNALS_DIR}"
        )
    endif()
endfunction()


# --------------------------------------------------------------------------
# Helper function for simple externals (no ONNX Runtime dependency)
# --------------------------------------------------------------------------
function(create_simple_external TARGET_NAME SOURCE_FILE)
    add_library(${TARGET_NAME} MODULE ${SOURCE_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${COMMON_INCLUDES})

    if(PLATFORM_WINDOWS)
        # Locate Max SDK libraries
        find_library(MAXAPI_LIB NAMES MaxAPI
                PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes/x64"
                NO_DEFAULT_PATH
        )
        find_library(MAXAUDIO_LIB NAMES MaxAudio
                PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/msp-includes/x64"
                NO_DEFAULT_PATH
        )
        find_library(JITTER_LIB NAMES jitlib
                PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/jit-includes/x64"
                NO_DEFAULT_PATH
        )

        if(NOT MAXAPI_LIB)
            message(FATAL_ERROR "Could not find MaxAPI.lib in Max SDK")
        endif()

        target_link_libraries(${TARGET_NAME} PRIVATE
                ${PLATFORM_LIBRARIES}
                ${MAXAPI_LIB}
                ${MAXAUDIO_LIB}    # optional (MSP)
                ${JITTER_LIB}      # optional (Jitter)
        )

        set_target_properties(${TARGET_NAME} PROPERTIES
                SUFFIX ".${EXTERNAL_EXTENSION}"
                OUTPUT_NAME ${TARGET_NAME}
        )

        configure_file(
                ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
                ${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}.manifest
                @ONLY
        )
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE ${PLATFORM_LIBRARIES})
        set_target_properties(${TARGET_NAME} PROPERTIES
                LINK_FLAGS "-undefined dynamic_lookup"
                BUNDLE TRUE
                BUNDLE_EXTENSION ${EXTERNAL_EXTENSION}
                OUTPUT_NAME ${TARGET_NAME}
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}/Contents/MacOS
                INSTALL_RPATH "@loader_path"
        )
        configure_file(
                ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
                ${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}/Contents/Info.plist
                @ONLY
        )
    endif()

    setup_external_deployment(${TARGET_NAME})
endfunction()

# --------------------------------------------------------------------------
# Helper function for ONNX externals
# --------------------------------------------------------------------------
function(create_onnx_external TARGET_NAME SOURCE_FILE)
    add_library(${TARGET_NAME} MODULE ${SOURCE_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${COMMON_INCLUDES})

    if(PLATFORM_WINDOWS)
        # Locate Max SDK libraries
        find_library(MAXAPI_LIB NAMES MaxAPI
                PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes/x64"
                NO_DEFAULT_PATH
        )
        find_library(MAXAUDIO_LIB NAMES MaxAudio
                PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/msp-includes/x64"
                NO_DEFAULT_PATH
        )
        find_library(JITTER_LIB NAMES jitlib
                PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/jit-includes/x64"
                NO_DEFAULT_PATH
        )

        if(NOT MAXAPI_LIB)
            message(FATAL_ERROR "Could not find MaxAPI.lib in Max SDK")
        endif()

        target_link_libraries(${TARGET_NAME} PRIVATE
                "${ONNX_LIB}"
                ${PLATFORM_LIBRARIES}
                ${MAXAPI_LIB}
                ${MAXAUDIO_LIB}    # optional (MSP)
                ${JITTER_LIB}      # optional (Jitter)
        )

        # ADD THIS SECTION: Fix round symbol conflict
        target_link_options(${TARGET_NAME} PRIVATE
                "/FORCE:MULTIPLE"     # Allow multiple symbol definitions
                "/IGNORE:4006"        # Ignore warnings about multiple definitions
        )

        set_target_properties(${TARGET_NAME} PROPERTIES
                SUFFIX ".${EXTERNAL_EXTENSION}"
                OUTPUT_NAME ${TARGET_NAME}
        )

        configure_file(
                ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
                ${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}.manifest
                @ONLY
        )
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE
                "${ONNX_LIB}"
                ${PLATFORM_LIBRARIES}
        )
        set_target_properties(${TARGET_NAME} PROPERTIES
                LINK_FLAGS "-undefined dynamic_lookup"
                BUNDLE TRUE
                BUNDLE_EXTENSION ${EXTERNAL_EXTENSION}
                OUTPUT_NAME ${TARGET_NAME}
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}/Contents/MacOS
                INSTALL_RPATH "@loader_path"
        )
        configure_file(
                ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
                ${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}/Contents/Info.plist
                @ONLY
        )
    endif()

    setup_external_deployment(${TARGET_NAME})
endfunction()



# ------------------------------------------------------------------------------
# Function to copy all model variants to resources directory
# ------------------------------------------------------------------------------
function(setup_model_resources)
    add_custom_target(CopyModelResources ALL)

    # Copy ONNX model files
    foreach(MODEL_VARIANT ${ALL_MODEL_VARIANTS})
        get_filename_component(MODEL_DIR ${MODEL_VARIANT} DIRECTORY)
        get_filename_component(MODEL_FILE ${MODEL_VARIANT} NAME)
        set(RESOURCE_MODEL_DIR "${MAX_RESOURCES_DIR}/${MODEL_DIR}")

        add_custom_command(TARGET CopyModelResources POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCE_MODEL_DIR}"
                COMMENT "Created resources directory for ${MODEL_VARIANT}"
        )

        # Copy the .onnx file
        if(EXISTS "${CMAKE_SOURCE_DIR}/${MODEL_VARIANT}")
            add_custom_command(TARGET CopyModelResources POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_SOURCE_DIR}/${MODEL_VARIANT}"
                    "${RESOURCE_MODEL_DIR}/${MODEL_FILE}"
                    COMMENT "Copied ${MODEL_VARIANT} to resources"
            )
        endif()
    endforeach()

    # Copy artifacts (codebooks, metadata) to resources/artifacts
    set(ARTIFACTS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/artifacts")
    set(ARTIFACTS_DEST_DIR "${MAX_RESOURCES_DIR}/artifacts")

    if(EXISTS "${ARTIFACTS_SOURCE_DIR}")
        add_custom_command(TARGET CopyModelResources POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${ARTIFACTS_DEST_DIR}"
                COMMENT "Created resources/artifacts directory"
        )

        # Find and copy all .json files from artifacts
        file(GLOB ARTIFACT_JSON_FILES "${ARTIFACTS_SOURCE_DIR}/*.json")
        foreach(JSON_FILE ${ARTIFACT_JSON_FILES})
            get_filename_component(JSON_FILENAME ${JSON_FILE} NAME)
            add_custom_command(TARGET CopyModelResources POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${JSON_FILE}"
                    "${ARTIFACTS_DEST_DIR}/${JSON_FILENAME}"
                    COMMENT "Copied ${JSON_FILENAME} to resources/artifacts"
            )
        endforeach()

        # Find and copy all .f16bin files from artifacts
        file(GLOB ARTIFACT_BIN_FILES "${ARTIFACTS_SOURCE_DIR}/*.f16bin")
        foreach(BIN_FILE ${ARTIFACT_BIN_FILES})
            get_filename_component(BIN_FILENAME ${BIN_FILE} NAME)
            add_custom_command(TARGET CopyModelResources POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${BIN_FILE}"
                    "${ARTIFACTS_DEST_DIR}/${BIN_FILENAME}"
                    COMMENT "Copied ${BIN_FILENAME} to resources/artifacts"
            )
        endforeach()
    else()
        message(WARNING "Artifacts directory not found: ${ARTIFACTS_SOURCE_DIR}")
    endif()
endfunction()


# ------------------------------------------------------------------------------
# ONNX-based Externals
# ------------------------------------------------------------------------------

# Mute Genre Encoder Target
set(WATERFILL_TARGET WaterFill)
create_onnx_external(${WATERFILL_TARGET} ${SRC_DIR}/WaterFill.cpp)

# ------------------------------------------------------------------------------
# Setup model resources deployment
# ------------------------------------------------------------------------------
setup_model_resources()

# Make sure resources are copied when building the externals
set(ALL_EXTERNALS
        ${WATERFILL_TARGET}
)

foreach(EXTERNAL_NAME ${ALL_EXTERNALS})
    add_dependencies(${EXTERNAL_NAME} CopyModelResources)
endforeach()

# ------------------------------------------------------------------------------
# Install targets (Optional)
# ------------------------------------------------------------------------------
foreach(EXTERNAL_NAME ${ALL_EXTERNALS})
    if(PLATFORM_WINDOWS)
        # On Windows, externals are single .mxe64 files
        install(FILES
                "${CMAKE_BINARY_DIR}/${EXTERNAL_NAME}.${EXTERNAL_EXTENSION}"
                DESTINATION "${CMAKE_SOURCE_DIR}/externals"
        )
    else()
        # On macOS, externals are bundle directories (.mxo)
        install(DIRECTORY
                "${CMAKE_BINARY_DIR}/${EXTERNAL_NAME}.${EXTERNAL_EXTENSION}"
                DESTINATION "${CMAKE_SOURCE_DIR}/externals"
        )
    endif()
endforeach()

# Install model resources to repo as well
foreach(MODEL_VARIANT ${ALL_MODEL_VARIANTS})
    if(EXISTS "${CMAKE_SOURCE_DIR}/models/${MODEL_VARIANT}/")
        install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/${MODEL_VARIANT}/"
                DESTINATION "${CMAKE_SOURCE_DIR}/resources/${MODEL_VARIANT}"
                FILES_MATCHING PATTERN "*.onnx" PATTERN "*.onnx.data"
        )
    endif()
endforeach()


# ------------------------------------------------------------------------------
# Group Target
# ------------------------------------------------------------------------------
add_custom_target(WaterFillRNNAll
        DEPENDS ${ALL_EXTERNALS} CopyModelResources
)

# ------------------------------------------------------------------------------
# Usage Information
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
if(PLATFORM_MACOS AND ENABLE_CODESIGNING)
    message(STATUS "  Codesigning: ${ENABLE_CODESIGNING}")
    if(CODESIGN_IDENTITY STREQUAL "")
        message(STATUS "  Identity: Ad-hoc")
    else()
        message(STATUS "  Identity: ${CODESIGN_IDENTITY}")
    endif()
endif()
message(STATUS "  External extension: ${EXTERNAL_EXTENSION}")
message(STATUS "")
message(STATUS "ONNX Runtime:")
message(STATUS "  Archive: ${ONNX_ARCHIVE_NAME}")
message(STATUS "  Library: ${ONNX_LIB}")
message(STATUS "  Static: ${ONNX_IS_STATIC}")
message(STATUS "")
message(STATUS "Package structure:")
message(STATUS "  ${MAX_PACKAGE_DIR}/")
message(STATUS "    externals/")
foreach(EXTERNAL_NAME ${ALL_EXTERNALS})
    message(STATUS "      ${EXTERNAL_NAME}.${EXTERNAL_EXTENSION}")
endforeach()
message(STATUS "    resources/")
foreach(MODEL_VARIANT ${ALL_MODEL_VARIANTS})
    message(STATUS "      ${MODEL_VARIANT}")
endforeach()
message(STATUS "      artifacts/")
message(STATUS "        *.json (codebook metadata)")
message(STATUS "        *.f16bin (codebook data)")
message(STATUS "")
message(STATUS "Available build targets:")
message(STATUS "  WaterFillRNNAll          - Build all externals and copy all model resources")
foreach(EXTERNAL_NAME ${ALL_EXTERNALS})
    message(STATUS "  ${EXTERNAL_NAME}")
endforeach()
message(STATUS "  CopyModelResources               - Copy model resources only")
message(STATUS "")
if(PLATFORM_MACOS)
    message(STATUS "To configure codesigning:")
    message(STATUS "  Ad-hoc:     cmake -DCODESIGN_IDENTITY=\"\" ...")
    message(STATUS "  Developer:  cmake -DCODESIGN_IDENTITY=\"Developer ID Application: Your Name (TEAMID)\" ...")
    message(STATUS "  Disable:    cmake -DENABLE_CODESIGNING=OFF ...")
endif()
message(STATUS "")