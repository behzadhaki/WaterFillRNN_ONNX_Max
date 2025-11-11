#pragma once

#include "c74_min.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>  // For SHGetFolderPath
#include <filesystem>
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#endif

#include <onnxruntime_cxx_api.h>
#include "ext.h"  // For Max externals
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <condition_variable>

#ifdef _WIN32
// Forward declare Jitter's round (exported from jitlib.dll).
extern "C" double round(double);

// Prevent accidental use of std::round.
#undef round
#endif

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#endif

// Enhanced Thread-safe ONNX Environment Manager with diagnostics
class ONNXManager {
private:
    // Use raw pointer instead of unique_ptr to avoid static destruction order issues
    // We intentionally leak this on program exit to avoid crashes
    static Ort::Env* env;
    static std::mutex env_mutex;
    static std::atomic<int> ref_count;
    static std::atomic<bool> is_shutting_down;
    static std::atomic<bool> env_valid;

    ONNXManager() = default;

public:
    static Ort::Env& get_env() {
        std::lock_guard<std::mutex> lock(env_mutex);

        if (is_shutting_down.load()) {
            throw std::runtime_error("ONNX environment is shutting down");
        }

        if (!env || !env_valid.load()) {
            // Create the environment - we'll intentionally leak this on exit
            // This is a common pattern for singletons that have complex destruction dependencies
            env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "shared_onnx_env");
            env_valid.store(true);

            // Register atexit handler to mark shutdown state
            std::atexit([]() {
                is_shutting_down.store(true);
                env_valid.store(false);
                // DO NOT delete env here - intentionally leak to avoid crashes
                // The OS will clean up the memory on process exit
            });
        }

        ref_count.fetch_add(1);
        return *env;
    }

    static void release_env() {
        std::lock_guard<std::mutex> lock(env_mutex);

        // Don't do anything if we're already shutting down
        if (is_shutting_down.load()) {
            return;
        }

        if (ref_count.load() <= 0) {
            return;
        }

        int new_count = ref_count.fetch_sub(1) - 1;

        // Never actually delete the environment - let it leak
        // This prevents crashes during shutdown
    }

    static void explicit_cleanup() {
        std::lock_guard<std::mutex> lock(env_mutex);

        if (is_shutting_down.load()) {
            return; // Already shutting down
        }

        is_shutting_down.store(true);

        if (env && env_valid.load()) {
            env_valid.store(false);

            // Only delete if explicitly requested and not during exit
            // In practice, we might still want to leak even here for safety
            delete env;
            env = nullptr;

            ref_count.store(0);
        }

        is_shutting_down.store(false);
    }

    static int get_ref_count() {
        return ref_count.load();
    }

    static bool is_available() {
        std::lock_guard<std::mutex> lock(env_mutex);
        return env && env_valid.load() && !is_shutting_down.load();
    }

    // No destructor needed since we're using raw pointers
};

// Static member definitions
inline Ort::Env* ONNXManager::env = nullptr;
inline std::mutex ONNXManager::env_mutex;
inline std::atomic<int> ONNXManager::ref_count(0);
inline std::atomic<bool> ONNXManager::is_shutting_down(false);
inline std::atomic<bool> ONNXManager::env_valid(false);

class BundleResourceLoader {
public:
    static std::string get_resource_path(const std::string& resource_name,
                                         const std::string& model_type,
                                         const std::string& beta_type = "beta_0.2") {
#ifdef _WIN32
        HMODULE hModule = NULL;
        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR)&get_resource_path, &hModule)) {
            wchar_t path[MAX_PATH];
            if (GetModuleFileNameW(hModule, path, MAX_PATH)) {
                std::wstring wpath(path);
                std::string external_path(wpath.begin(), wpath.end());

                // external_path = ".../{MAXPACKAGE}/externals/gt.baseEncoder.mxe64"
                size_t externals_pos = external_path.find("\\externals\\");
                if (externals_pos != std::string::npos) {
                    std::string package_path = external_path.substr(0, externals_pos);
                    std::string resources_path = package_path + "\\resources\\" +
                                                 model_type + "\\" + beta_type + "\\" +
                                                 resource_name;

                    if (file_exists(resources_path)) {
                        return resources_path;
                    }

                    // fallback: try other beta variants
                    std::vector<std::string> variants = {
                            model_type + "\\beta_0.2",
                            model_type + "\\beta_0.5",
                            model_type + "\\beta_1.0"
                    };

                    auto requested_variant = model_type + "\\" + beta_type;
                    for (const auto& variant : variants) {
                        if (variant == requested_variant) continue;
                        std::string alt_path = package_path + "\\resources\\" +
                                               variant + "\\" + resource_name;
                        if (file_exists(alt_path)) {
                            return alt_path;
                        }
                    }
                }
            }
        }
        return "";
#else
        // Get the address of this function to find which bundle we're in
        Dl_info info;
        if (dladdr((void*)get_resource_path, &info) != 0 && info.dli_fname) {
            std::string external_path(info.dli_fname);
            // Path should be something like: .../{MAXPACKAGE}/externals/gt.baseEncoder.mxo/Contents/MacOS/gt.baseEncoder
            // We want: .../{MAXPACKAGE}/resources/BaseVAE/beta_0.2/encoder.onnx

            // Find the package directory by going up from the external
            size_t externals_pos = external_path.find("/externals/");
            if (externals_pos != std::string::npos) {
                // Get the package root directory
                std::string package_path = external_path.substr(0, externals_pos);
                std::string resources_path = package_path + "/resources/" + model_type + "/" + beta_type + "/" + resource_name;

                if (file_exists(resources_path)) {
                    return resources_path;
                }

                // If the specific variant doesn't exist, try to find any available variant
                if (!file_exists(resources_path)) {
                    auto model_variant = model_type + "/" + beta_type;
                    // Try other common variants
                    std::vector<std::string> variants = {
                        model_type + "/" + "beta_0.2",
                        model_type + "/" + "beta_0.5",
                        model_type + "/" + "beta_1.0"
                    };

                    for (const auto& variant : variants) {
                        if (variant == model_variant) continue; // Already tried this one
                        std::string alt_path = package_path + "/resources/" + variant + "/" + resource_name;
                        if (file_exists(alt_path)) {
                            // Log that we're falling back to a different variant
                            return alt_path;
                        }
                    }
                }
            }
        }

        // Fallback: try the old bundle resource method for backward compatibility
        CFArrayRef bundles = CFBundleGetAllBundles();
        if (bundles) {
            CFIndex count = CFArrayGetCount(bundles);
            for (CFIndex i = 0; i < count; i++) {
                CFBundleRef bundle = (CFBundleRef)CFArrayGetValueAtIndex(bundles, i);
                if (bundle) {
                    // Check if this bundle has our resource
                    CFStringRef resource_cf = CFStringCreateWithCString(kCFAllocatorDefault,
                                                                      resource_name.c_str(),
                                                                      kCFStringEncodingUTF8);
                    if (resource_cf) {
                        CFURLRef resource_url = CFBundleCopyResourceURL(bundle, resource_cf, nullptr, nullptr);
                        CFRelease(resource_cf);

                        if (resource_url) {
                            char path_buffer[2048];
                            if (CFURLGetFileSystemRepresentation(resource_url, true,
                                                               (UInt8*)path_buffer,
                                                               sizeof(path_buffer))) {
                                CFRelease(resource_url);

                                // Verify the file exists before returning
                                if (file_exists(std::string(path_buffer))) {
                                    return std::string(path_buffer);
                                }
                            }
                            CFRelease(resource_url);
                        }
                    }
                }
            }
        }
#endif
        return "";
    }

    static std::string get_package_resources_path() {
#ifdef _WIN32
        HMODULE hModule = NULL;
        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR)&get_package_resources_path, &hModule)) {
            wchar_t path[MAX_PATH];
            if (GetModuleFileNameW(hModule, path, MAX_PATH)) {
                std::wstring wpath(path);
                std::string external_path(wpath.begin(), wpath.end());
                size_t externals_pos = external_path.find("\\externals\\");
                if (externals_pos != std::string::npos) {
                    std::string package_path = external_path.substr(0, externals_pos);
                    return package_path + "\\resources";
                }
            }
        }
        return "";
#else
        Dl_info info;
        if (dladdr((void*)get_resource_path, &info) != 0 && info.dli_fname) {
            std::string external_path(info.dli_fname);
            size_t externals_pos = external_path.find("/externals/");
            if (externals_pos != std::string::npos) {
                std::string package_path = external_path.substr(0, externals_pos);
                return package_path + "/resources";
            }
        }
#endif
    }

    static bool file_exists(const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    }
};