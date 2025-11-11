// gt.waterFillRNN.cpp
// Max/MSP external for real-time RNN-based audio generation
// Background thread handles ONNX inference, main thread handles audio I/O

#include "c74_min.h"
#include "shared_external_helpers.h"
#include "onnx_helpers.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace c74::min;

// ============================================================================
// Constants
// ============================================================================
constexpr int NQ = 8;                          // Number of codebook quantizers
constexpr int CHUNK_FRAMES = 32;               // Decoder FIFO length
constexpr int HOP_FRAMES = 8;                  // RNN steps per hop
constexpr int SAMPLES_PER_FRAME_24K = 320;     // Samples per frame at 24kHz
constexpr int HOP_SAMPLES_24K = HOP_FRAMES * SAMPLES_PER_FRAME_24K; // 2560
constexpr int LATENT_DIM = 128;
constexpr int HIDDEN_SIZE = 128;
constexpr int NUM_LAYERS = 3;
constexpr int CODEBOOK_SIZE = 1024;
constexpr int COND_DIM = 1;
constexpr float CLAMP_VAL = 15.0f;

// ============================================================================
// Message Structures for Thread Communication
// ============================================================================

enum class MessageType {
    REQUEST_HOP,    // Main -> Worker: request new audio hop
    AUDIO_HOP,      // Worker -> Main: deliver audio samples
    SHUTDOWN,       // Main -> Worker: clean shutdown
    ERROR           // Worker -> Main: error occurred
};

struct Message {
    MessageType type;
    float conditioning_value;  // For REQUEST_HOP
    std::vector<float> audio_data;  // For AUDIO_HOP
    std::string error_msg;     // For ERROR
};

// Thread-safe queue
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;

public:
    void push(const T& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(item);
        }
        cond_.notify_one();
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool wait_pop(T& item, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                           [this] { return !queue_.empty(); })) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T>().swap(queue_);
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

// Float16 to Float32 conversion
inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000u) >> 15;
    uint32_t e = (h & 0x7C00u) >> 10;
    uint32_t f = h & 0x03FFu;

    if (e == 0) {
        return (s ? -1.0f : 1.0f) * std::pow(2.0f, -14.0f) * (f / 1024.0f);
    }
    if (e == 0x1F) {
        return f ? NAN : (s ? -1.0f : 1.0f) * INFINITY;
    }
    return (s ? -1.0f : 1.0f) * std::pow(2.0f, (float)e - 15.0f) * (1.0f + f / 1024.0f);
}

// Clamp and scale latents to [-1, 1]
inline void clamp_and_scale_128(const float* in, float* out) {
    for (int i = 0; i < LATENT_DIM; i++) {
        float v = in[i];
        v = std::max(-CLAMP_VAL, std::min(CLAMP_VAL, v));
        out[i] = v / CLAMP_VAL;
    }
}

// Simple linear upsampling 24kHz -> 48kHz
std::vector<float> upsample_2x_linear(const float* input, size_t input_len) {
    std::vector<float> output(input_len * 2);
    for (size_t i = 0; i < input_len - 1; i++) {
        output[i * 2] = input[i];
        output[i * 2 + 1] = (input[i] + input[i + 1]) * 0.5f;
    }
    output[(input_len - 1) * 2] = input[input_len - 1];
    output[(input_len - 1) * 2 + 1] = input[input_len - 1];
    return output;
}

// Top-k sampling from logits
int sample_from_logits(const float* logits, int K, float temperature, int top_k) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (temperature <= 0.0f || top_k == 1) {
        // Argmax
        return std::distance(logits, std::max_element(logits, logits + K));
    }

    // Create pairs for top-k selection
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(K);
    for (int i = 0; i < K; i++) {
        pairs.push_back({logits[i], i});
    }

    // Sort and keep top-k
    std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    // Softmax with temperature on top-k
    float max_logit = pairs[0].first;
    std::vector<float> probs(top_k);
    float sum = 0.0f;

    for (int i = 0; i < top_k; i++) {
        float exp_val = std::exp((pairs[i].first - max_logit) / temperature);
        probs[i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < top_k; i++) {
        probs[i] /= sum;
    }

    // Sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(gen);
    float acc = 0.0f;

    for (int i = 0; i < top_k; i++) {
        acc += probs[i];
        if (r <= acc) {
            return pairs[i].second;
        }
    }

    return pairs[top_k - 1].second;
}

// ============================================================================
// Codebook Table Loader
// ============================================================================
struct CodebookTable {
    std::vector<float> data;  // Shape: [n_q, K, D]
    int n_q = NQ;
    int K = CODEBOOK_SIZE;
    int D = LATENT_DIM;

    bool load_from_file(const std::string& meta_path, const std::string& bin_path) {
        try {
            // Load metadata JSON
            std::ifstream meta_file(meta_path);
            if (!meta_file) {
                std::cerr << "Failed to open codebook meta: " << meta_path << std::endl;
                return false;
            }

            // Simple JSON parsing for our known structure
            std::string line;
            while (std::getline(meta_file, line)) {
                // Look for n_q, num_codebooks, or nq
                if (line.find("\"n_q\"") != std::string::npos ||
                    line.find("\"num_codebooks\"") != std::string::npos ||
                    line.find("\"nq\"") != std::string::npos) {
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string value = line.substr(colon + 1);
                        n_q = std::stoi(value);
                    }
                }
                // Look for K, codebook_size, or num_entries
                if (line.find("\"codebook_size\"") != std::string::npos ||
                    line.find("\"K\"") != std::string::npos ||
                    line.find("\"num_entries\"") != std::string::npos) {
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string value = line.substr(colon + 1);
                        K = std::stoi(value);
                    }
                }
                // Look for D, dim, latent_dim, or embedding_dim
                if (line.find("\"dim\"") != std::string::npos ||
                    line.find("\"D\"") != std::string::npos ||
                    line.find("\"latent_dim\"") != std::string::npos ||
                    line.find("\"embedding_dim\"") != std::string::npos) {
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string value = line.substr(colon + 1);
                        D = std::stoi(value);
                    }
                }
            }
            meta_file.close();

            std::cout << "Codebook meta: n_q=" << n_q << ", K=" << K << ", D=" << D << std::endl;

            // Load binary f16 data
            std::ifstream file(bin_path, std::ios::binary);
            if (!file) {
                std::cerr << "Failed to open codebook: " << bin_path << std::endl;
                return false;
            }

            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            size_t expected_size = n_q * K * D * sizeof(uint16_t);
            if (file_size != expected_size) {
                std::cerr << "Codebook size mismatch: " << file_size
                         << " vs " << expected_size << std::endl;
                // Try to proceed anyway with available data
            }

            size_t elements = file_size / sizeof(uint16_t);
            std::vector<uint16_t> f16_data(elements);
            file.read(reinterpret_cast<char*>(f16_data.data()), file_size);
            file.close();

            // Convert to float32
            data.resize(elements);
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = f16_to_f32(f16_data[i]);
            }

            std::cout << "Loaded codebook: " << n_q << " x " << K << " x " << D << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Codebook load error: " << e.what() << std::endl;
            return false;
        }
    }

    void get_latent(int q, int idx, float* out) const {
        int base = q * (K * D) + idx * D;
        if (base + D > static_cast<int>(data.size())) {
            std::fill(out, out + D, 0.0f);
            return;
        }
        std::copy(data.begin() + base, data.begin() + base + D, out);
    }
};

// ============================================================================
// Background Worker Thread
// ============================================================================
class RNNWorker {
private:
    Ort::Env* env_;  // Use pointer to shared environment
    std::unique_ptr<Ort::Session> rnn_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    std::unique_ptr<CodebookTable> codebook_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // State
    std::vector<float> last_hidden_;     // [NUM_LAYERS, 1, HIDDEN_SIZE]
    std::vector<float> last_latent_;     // [LATENT_DIM]
    std::vector<int64_t> code_fifo_;     // [NQ, CHUNK_FRAMES]
    int fifo_write_pos_ = 0;

    // Sampling params
    float temperature_ = 1.0f;
    int top_k_ = 8;

    // Thread control
    std::thread worker_thread_;
    std::atomic<bool> running_{false};

    // Queues
    ThreadSafeQueue<Message> input_queue_;
    ThreadSafeQueue<Message> output_queue_;

public:
    RNNWorker() : env_(nullptr) {}

    ~RNNWorker() {
        shutdown();
        if (env_) {
            ONNXManager::release_env();
        }
    }

    bool initialize(const std::string& resources_path) {
        try {
            std::cout << "[RNNWorker] Starting initialization..." << std::endl;

            // Get shared ONNX environment
            env_ = &ONNXManager::get_env();
            std::cout << "[RNNWorker] ✓ ONNX environment acquired" << std::endl;

            // Session options
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            // Load RNN model using BundleResourceLoader
            std::string rnn_path = BundleResourceLoader::get_resource_path(
                "rnn_step.onnx", "onnx");

            if (rnn_path.empty() || !BundleResourceLoader::file_exists(rnn_path)) {
                // Fallback to manual path construction
                rnn_path = resources_path + "/onnx/rnn_step.onnx";
            }

            std::cout << "[RNNWorker] Loading RNN model from: " << rnn_path << std::endl;

            if (!BundleResourceLoader::file_exists(rnn_path)) {
                std::cerr << "[RNNWorker] ERROR: RNN model file not found!" << std::endl;
                return false;
            }

            rnn_session_ = std::make_unique<Ort::Session>(*env_, rnn_path.c_str(), session_options);
            std::cout << "[RNNWorker] ✓ RNN model loaded successfully" << std::endl;

            // Load decoder model
            std::string decoder_path = BundleResourceLoader::get_resource_path(
                "encodec_decode.onnx", "onnx");

            if (decoder_path.empty() || !BundleResourceLoader::file_exists(decoder_path)) {
                decoder_path = resources_path + "/onnx/encodec_decode.onnx";
            }

            std::cout << "[RNNWorker] Loading decoder model from: " << decoder_path << std::endl;

            if (!BundleResourceLoader::file_exists(decoder_path)) {
                std::cerr << "[RNNWorker] ERROR: Decoder model file not found!" << std::endl;
                return false;
            }

            decoder_session_ = std::make_unique<Ort::Session>(*env_, decoder_path.c_str(), session_options);
            std::cout << "[RNNWorker] ✓ Decoder model loaded successfully" << std::endl;

            // Load codebook
            codebook_ = std::make_unique<CodebookTable>();

            std::string meta_path = BundleResourceLoader::get_resource_path(
                "encodec24_codebooks.meta.json", "artifacts");
            std::string bin_path = BundleResourceLoader::get_resource_path(
                "encodec24_codebooks.f16bin", "artifacts");

            if (meta_path.empty() || !BundleResourceLoader::file_exists(meta_path)) {
                meta_path = resources_path + "/artifacts/encodec24_codebooks.meta.json";
            }
            if (bin_path.empty() || !BundleResourceLoader::file_exists(bin_path)) {
                bin_path = resources_path + "/artifacts/encodec24_codebooks.f16bin";
            }

            std::cout << "[RNNWorker] Loading codebook metadata from: " << meta_path << std::endl;
            std::cout << "[RNNWorker] Loading codebook data from: " << bin_path << std::endl;

            if (!BundleResourceLoader::file_exists(meta_path)) {
                std::cerr << "[RNNWorker] ERROR: Codebook metadata file not found!" << std::endl;
                return false;
            }

            if (!BundleResourceLoader::file_exists(bin_path)) {
                std::cerr << "[RNNWorker] ERROR: Codebook binary file not found!" << std::endl;
                return false;
            }

            if (!codebook_->load_from_file(meta_path, bin_path)) {
                std::cerr << "[RNNWorker] ERROR: Failed to load codebook!" << std::endl;
                return false;
            }

            std::cout << "[RNNWorker] ✓ Codebook loaded successfully" << std::endl;

            // Initialize state
            last_hidden_.resize(NUM_LAYERS * 1 * HIDDEN_SIZE, 0.0f);
            last_latent_.resize(LATENT_DIM);

            // Random seed for latent
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
            for (auto& v : last_latent_) {
                v = dist(gen);
            }

            // Initialize code FIFO
            code_fifo_.resize(NQ * CHUNK_FRAMES, 0);

            std::cout << "[RNNWorker] ✓ State initialized" << std::endl;
            std::cout << "[RNNWorker] ========================================" << std::endl;
            std::cout << "[RNNWorker] INITIALIZATION COMPLETE - ALL RESOURCES LOADED" << std::endl;
            std::cout << "[RNNWorker] ========================================" << std::endl;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "[RNNWorker] INITIALIZATION FAILED: " << e.what() << std::endl;
            return false;
        }
    }

    void start() {
        if (running_) return;
        running_ = true;
        worker_thread_ = std::thread(&RNNWorker::worker_loop, this);
    }

    void shutdown() {
        if (!running_) return;

        running_ = false;

        // Send shutdown message
        Message msg;
        msg.type = MessageType::SHUTDOWN;
        input_queue_.push(msg);

        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

        input_queue_.clear();
        output_queue_.clear();
    }

    void request_hop(float conditioning) {
        Message msg;
        msg.type = MessageType::REQUEST_HOP;
        msg.conditioning_value = conditioning;
        input_queue_.push(msg);
    }

    bool get_output_message(Message& msg) {
        return output_queue_.try_pop(msg);
    }

    size_t pending_requests() const {
        return input_queue_.size();
    }

private:
    void worker_loop() {
        while (running_) {
            Message msg;

            // Wait for incoming message
            if (!input_queue_.wait_pop(msg, 100)) {
                continue;
            }

            if (msg.type == MessageType::SHUTDOWN) {
                break;
            }

            if (msg.type == MessageType::REQUEST_HOP) {
                process_hop_request(msg.conditioning_value);
            }
        }

        std::cout << "Worker thread exiting" << std::endl;
    }

    void process_hop_request(float conditioning) {
        try {
            // Generate HOP_FRAMES worth of codes
            for (int hop = 0; hop < HOP_FRAMES; hop++) {
                std::vector<int> codes = rnn_step(conditioning);
                push_codes_to_fifo(codes);
            }

            // Decode the full chunk
            std::vector<float> audio_24k = decode_chunk();

            // Extract tail (last HOP_FRAMES worth)
            size_t tail_start = audio_24k.size() - HOP_SAMPLES_24K;
            std::vector<float> tail_24k(audio_24k.begin() + tail_start, audio_24k.end());

            // Upsample to 48kHz
            std::vector<float> audio_48k = upsample_2x_linear(tail_24k.data(), tail_24k.size());

            // Send result back
            Message result;
            result.type = MessageType::AUDIO_HOP;
            result.audio_data = std::move(audio_48k);
            output_queue_.push(result);

        } catch (const std::exception& e) {
            Message error;
            error.type = MessageType::ERROR;
            error.error_msg = std::string("Processing error: ") + e.what();
            output_queue_.push(error);
        }
    }

    std::vector<int> rnn_step(float conditioning) {
        // Prepare inputs
        std::vector<float> cond_vec = {conditioning};

        std::vector<int64_t> latent_shape = {1, LATENT_DIM};
        std::vector<int64_t> cond_shape = {1, COND_DIM};
        std::vector<int64_t> hidden_shape = {NUM_LAYERS, 1, HIDDEN_SIZE};

        // Use helper functions from onnx_helpers.h
        Ort::Value latent_tensor = vector_to_tensor(last_latent_, latent_shape, allocator_);
        Ort::Value cond_tensor = vector_to_tensor(cond_vec, cond_shape, allocator_);
        Ort::Value hidden_tensor = vector_to_tensor(last_hidden_, hidden_shape, allocator_);

        // Input names
        const char* input_names[] = {"latent_in", "cond_in", "hidden_in"};
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(latent_tensor));
        inputs.push_back(std::move(cond_tensor));
        inputs.push_back(std::move(hidden_tensor));

        // Output names
        const char* output_names[] = {"logits_out", "hidden_out", "step_latent"};

        // Run inference
        auto outputs = rnn_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, inputs.data(), inputs.size(),
            output_names, 3);

        // Extract outputs
        float* logits_ptr = outputs[0].GetTensorMutableData<float>();
        float* hidden_out_ptr = outputs[1].GetTensorMutableData<float>();

        // Update hidden state
        std::copy(hidden_out_ptr, hidden_out_ptr + last_hidden_.size(), last_hidden_.begin());

        // Sample codes and build new latent
        std::vector<int> sampled_codes(NQ);
        std::vector<float> new_latent(LATENT_DIM, 0.0f);
        std::vector<float> latent_q(LATENT_DIM);

        for (int q = 0; q < NQ; q++) {
            const float* logits_q = logits_ptr + q * CODEBOOK_SIZE;
            int idx = sample_from_logits(logits_q, CODEBOOK_SIZE, temperature_, top_k_);
            sampled_codes[q] = idx;

            codebook_->get_latent(q, idx, latent_q.data());
            for (int d = 0; d < LATENT_DIM; d++) {
                new_latent[d] += latent_q[d];
            }
        }

        // Clamp and scale for next iteration
        clamp_and_scale_128(new_latent.data(), last_latent_.data());

        return sampled_codes;
    }

    void push_codes_to_fifo(const std::vector<int>& codes) {
        for (int q = 0; q < NQ; q++) {
            code_fifo_[q * CHUNK_FRAMES + fifo_write_pos_] = codes[q];
        }
        fifo_write_pos_ = (fifo_write_pos_ + 1) % CHUNK_FRAMES;
    }

    std::vector<float> decode_chunk() {
        // Build input tensor [1, NQ, CHUNK_FRAMES] in q-major, time order
        std::vector<int64_t> codes_3d(1 * NQ * CHUNK_FRAMES);

        for (int q = 0; q < NQ; q++) {
            for (int t = 0; t < CHUNK_FRAMES; t++) {
                int pos = (fifo_write_pos_ + t) % CHUNK_FRAMES;
                int64_t code = code_fifo_[q * CHUNK_FRAMES + pos];
                codes_3d[q * CHUNK_FRAMES + t] = code;
            }
        }

        std::vector<int64_t> codes_shape = {1, NQ, CHUNK_FRAMES};

        // Use helper function from onnx_helpers.h
        Ort::Value codes_tensor = vector_to_tensor_i64(codes_3d, codes_shape, allocator_);

        const char* input_names[] = {"codes_bnt"};
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(codes_tensor));

        const char* output_names[] = {"audio"};

        auto outputs = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, inputs.data(), inputs.size(),
            output_names, 1);

        // Use helper function to extract tensor data
        return tensor_to_vector(outputs[0]);
    }
};

// ============================================================================
// Max/MSP External Class
// ============================================================================
class waterfill_rnn : public object<waterfill_rnn>, public sample_operator<0, 1> {
private:
    std::unique_ptr<RNNWorker> worker_;

    // Circular audio buffer (ring buffer for generated audio)
    static constexpr size_t RING_BUFFER_SIZE = 10240;  // ~16 RNN steps at 48kHz
    std::vector<float> ring_buffer_;
    size_t write_ptr_ = 0;
    size_t read_ptr_ = 0;
    size_t available_ = 0;

    // Flow control
    size_t low_water_mark_ = RING_BUFFER_SIZE / 2;
    bool hop_requested_ = false;

    // State
    std::atomic<bool> active_{false};
    std::atomic<bool> initialized_{false};
    float conditioning_value_ = 0.5f;

    // Max path
    std::string resources_path_;

public:
    MIN_DESCRIPTION{"Real-time RNN-based audio generator"};
    MIN_TAGS{"audio, generator, ai, rnn"};
    MIN_AUTHOR{"Cycling '74"};
    MIN_RELATED{"gt.muteGenreLatentEncoder"};

    outlet<> output{this, "(signal) Generated audio", "signal"};
    outlet<> status_out{this, "(message) Status messages"};

    // Attributes
    attribute<number> fillLevel{
        this, "fillLevel", 0.5,
        range{0.0, 1.0},
        description{"Conditioning parameter (0.0-1.0)"},
        setter{MIN_FUNCTION{
            conditioning_value_ = args[0];
            return args;
        }}
    };

    attribute<bool> active{
        this, "active", false,
        description{"Enable/disable audio generation"},
        setter{MIN_FUNCTION{
            active_ = (bool)args[0];
            if (!active_) {
                // Clear buffer when stopping
                clear_ring_buffer();
            }
            return args;
        }}
    };

    // Constructor
    waterfill_rnn(const atoms& args = {}) {
        ring_buffer_.resize(RING_BUFFER_SIZE, 0.0f);

        // Get resources path using helper
        resources_path_ = BundleResourceLoader::get_package_resources_path();

        if (resources_path_.empty()) {
            cerr << "Warning: Could not locate resources folder" << c74::min::endl;
            return;
        }

        cout << "Found resources at: " << resources_path_ << c74::min::endl;
        cout << "Checking required files..." << c74::min::endl;

        // Check for ONNX models
        std::string rnn_path = BundleResourceLoader::get_resource_path("rnn_step.onnx", "onnx");
        if (rnn_path.empty() || !BundleResourceLoader::file_exists(rnn_path)) {
            rnn_path = resources_path_ + "/onnx/rnn_step.onnx";
        }

        if (BundleResourceLoader::file_exists(rnn_path)) {
            cout << "✓ Found rnn_step.onnx at: " << rnn_path << c74::min::endl;
        } else {
            cerr << "✗ MISSING: rnn_step.onnx (expected at: " << rnn_path << ")" << c74::min::endl;
        }

        std::string decoder_path = BundleResourceLoader::get_resource_path("encodec_decode.onnx", "onnx");
        if (decoder_path.empty() || !BundleResourceLoader::file_exists(decoder_path)) {
            decoder_path = resources_path_ + "/onnx/encodec_decode.onnx";
        }

        if (BundleResourceLoader::file_exists(decoder_path)) {
            cout << "✓ Found encodec_decode.onnx at: " << decoder_path << c74::min::endl;
        } else {
            cerr << "✗ MISSING: encodec_decode.onnx (expected at: " << decoder_path << ")" << c74::min::endl;
        }

        // Check for artifacts (codebook files)
        std::string meta_path = BundleResourceLoader::get_resource_path("encodec24_codebooks.meta.json", "artifacts");
        if (meta_path.empty() || !BundleResourceLoader::file_exists(meta_path)) {
            meta_path = resources_path_ + "/artifacts/encodec24_codebooks.meta.json";
        }

        if (BundleResourceLoader::file_exists(meta_path)) {
            cout << "✓ Found encodec24_codebooks.meta.json at: " << meta_path << c74::min::endl;
        } else {
            cerr << "✗ MISSING: encodec24_codebooks.meta.json (expected at: " << meta_path << ")" << c74::min::endl;
        }

        std::string bin_path = BundleResourceLoader::get_resource_path("encodec24_codebooks.f16bin", "artifacts");
        if (bin_path.empty() || !BundleResourceLoader::file_exists(bin_path)) {
            bin_path = resources_path_ + "/artifacts/encodec24_codebooks.f16bin";
        }

        if (BundleResourceLoader::file_exists(bin_path)) {
            cout << "✓ Found encodec24_codebooks.f16bin at: " << bin_path << c74::min::endl;
        } else {
            cerr << "✗ MISSING: encodec24_codebooks.f16bin (expected at: " << bin_path << ")" << c74::min::endl;
        }

        cout << "Resource check complete." << c74::min::endl;
    }

    // Destructor
    ~waterfill_rnn() {
        if (worker_) {
            worker_->shutdown();
        }
    }

    // Initialize message
    message<> loadbang{
        this, "loadbang",
        MIN_FUNCTION{
            initialize_worker();
            return {};
        }
    };

    // Manual initialize
    message<> initialize{
        this, "initialize",
        MIN_FUNCTION{
            initialize_worker();
            return {};
        }
    };

    // DSP setup
    message<> dspsetup{
        this, "dspsetup",
        MIN_FUNCTION{
            cout << "DSP setup: samplerate = " << samplerate() << c74::min::endl;

            // Re-initialize worker if sample rate changed
            if (initialized_) {
                initialize_worker();
            }

            return {};
        }
    };

    // Audio processing (called per sample)
    sample operator()() {
        if (!active_ || !initialized_) {
            return 0.0;
        }

        // Check for completed audio hops from worker
        process_worker_messages();

        // Read from ring buffer
        float output_sample = 0.0f;
        if (available_ > 0) {
            output_sample = ring_buffer_[read_ptr_];
            read_ptr_ = (read_ptr_ + 1) % RING_BUFFER_SIZE;
            available_--;
        }

        // Request new hop if running low
        if (!hop_requested_ && available_ < low_water_mark_) {
            request_audio_hop();
        }

        return output_sample;
    }

private:
    void initialize_worker() {
        if (resources_path_.empty()) {
            status_out.send("error", "Resources path not found");
            return;
        }

        // Shutdown existing worker
        if (worker_) {
            worker_->shutdown();
        }

        // Create and initialize new worker
        worker_ = std::make_unique<RNNWorker>();

        if (worker_->initialize(resources_path_)) {
            worker_->start();
            initialized_ = true;

            // Pre-fill with first hop
            request_audio_hop();

            status_out.send("ready");
            cout << "WaterFill RNN initialized successfully" << c74::min::endl;
        } else {
            worker_.reset();
            initialized_ = false;
            status_out.send("error", "Failed to initialize RNN worker");
            cerr << "Failed to initialize RNN worker" << c74::min::endl;
        }
    }

    void request_audio_hop() {
        if (!worker_ || !initialized_) return;

        hop_requested_ = true;
        worker_->request_hop(conditioning_value_);
    }

    void process_worker_messages() {
        if (!worker_) return;

        Message msg;
        while (worker_->get_output_message(msg)) {
            switch (msg.type) {
                case MessageType::AUDIO_HOP:
                    receive_audio_hop(msg.audio_data);
                    break;

                case MessageType::ERROR:
                    status_out.send("error", msg.error_msg);
                    cerr << "Worker error: " << msg.error_msg << c74::min::endl;
                    break;

                default:
                    break;
            }
        }
    }

    void receive_audio_hop(const std::vector<float>& audio) {
        hop_requested_ = false;

        // Push audio into ring buffer
        for (float sample : audio) {
            ring_buffer_[write_ptr_] = sample;
            write_ptr_ = (write_ptr_ + 1) % RING_BUFFER_SIZE;

            if (available_ < RING_BUFFER_SIZE) {
                available_++;
            } else {
                // Buffer full, overwrite oldest
                read_ptr_ = (read_ptr_ + 1) % RING_BUFFER_SIZE;
            }
        }
    }

    void clear_ring_buffer() {
        std::fill(ring_buffer_.begin(), ring_buffer_.end(), 0.0f);
        write_ptr_ = 0;
        read_ptr_ = 0;
        available_ = 0;
        hop_requested_ = false;
    }
};

MIN_EXTERNAL(waterfill_rnn);