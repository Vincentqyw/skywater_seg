// Sky-Water-Person Segmentation — C++ ONNX Runtime inference
//
// Uses stb_image for reading/writing images (no OpenCV required).
// Reference: ONNX Runtime inference examples (MNIST, fns_candy_style_transfer).
//
// Build:
//   1. Download ONNX Runtime SDK from https://github.com/microsoft/onnxruntime/releases
//   2. cmake -B build -DONNXRUNTIME_ROOTDIR=<path-to-ort-sdk>
//   3. cmake --build build --config Release
//
// Usage:
//   skywater_infer.exe <model.onnx> <input.jpg> <output.png> [cpu|cuda|dml] [--iters 100]

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════
// Constants — must match training config
// ═══════════════════════════════════════════════════════════════════════

constexpr int INPUT_H = 384;
constexpr int INPUT_W = 384;
constexpr int NUM_CLASSES = 4;            // bg, sky, water, person
constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

// Class colours (RGB) — matches skywater_seg/visualization.py CLASS_COLORS_RGB
constexpr uint8_t CLASS_COLORS[NUM_CLASSES][3] = {
    {0, 0, 0},        // background: black
    {255, 140, 0},    // sky:        orange
    {0, 200, 255},    // water:      cyan
    {255, 60, 60},    // person:     red
};

// ═══════════════════════════════════════════════════════════════════════
// FP16 ↔ FP32 conversion
// ═══════════════════════════════════════════════════════════════════════

// IEEE 754 half-precision float → single-precision float
static float half_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  const uint32_t exp_raw = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;

  if (exp_raw == 0) {
    // Zero / subnormal
    if (mant == 0) {
      uint32_t r = sign;
      float f;
      std::memcpy(&f, &r, sizeof(f));
      return f;
    }
    // Normalise subnormal
    uint32_t m = mant;
    int e = 1;
    while ((m & 0x400u) == 0) { m <<= 1; --e; }
    uint32_t bits = sign | ((127 - 15 + e) << 23) | ((m & 0x3FFu) << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }
  if (exp_raw == 31) {
    // Inf / NaN
    uint32_t bits = sign | 0x7F800000u | (mant << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }
  // Normal number
  uint32_t bits = sign | ((exp_raw - 15 + 127) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// 64K-entry lookup table — turns per-element branchy conversion into
// a single array lookup (256 KB, negligible vs the image buffers).
static const auto kHalfToFloatLUT = []() {
  std::array<float, 65536> lut{};
  for (uint32_t i = 0; i < 65536; ++i)
    lut[i] = half_to_float(static_cast<uint16_t>(i));
  return lut;
}();

// ═══════════════════════════════════════════════════════════════════════
// Bilinear resize (CHW layout, contiguous float)
// ═══════════════════════════════════════════════════════════════════════

static void bilinear_resize(const float* src, int src_h, int src_w,
                            float* dst, int dst_h, int dst_w,
                            int channels) {
  const float scale_h = static_cast<float>(src_h) / dst_h;
  const float scale_w = static_cast<float>(src_w) / dst_w;
  const int src_stride = src_h * src_w;

  for (int c = 0; c < channels; ++c) {
    const float* ch_src = src + c * src_stride;
    float* ch_dst = dst + c * dst_h * dst_w;
    for (int dy = 0; dy < dst_h; ++dy) {
      const float sy = dy * scale_h;
      const int sy0 = static_cast<int>(sy);
      const int sy1 = std::min(sy0 + 1, src_h - 1);
      const float fy = sy - sy0;
      for (int dx = 0; dx < dst_w; ++dx) {
        const float sx = dx * scale_w;
        const int sx0 = static_cast<int>(sx);
        const int sx1 = std::min(sx0 + 1, src_w - 1);
        const float fx = sx - sx0;

        const float v00 = ch_src[sy0 * src_w + sx0];
        const float v01 = ch_src[sy0 * src_w + sx1];
        const float v10 = ch_src[sy1 * src_w + sx0];
        const float v11 = ch_src[sy1 * src_w + sx1];

        const float v0 = v00 + (v01 - v00) * fx;
        const float v1 = v10 + (v11 - v10) * fx;
        ch_dst[dy * dst_w + dx] = v0 + (v1 - v0) * fy;
      }
    }
  }
}

// Nearest-neighbour resize for uint8 masks
static void nearest_resize_uint8(const uint8_t* src, int src_h, int src_w,
                                 uint8_t* dst, int dst_h, int dst_w) {
  const float scale_h = static_cast<float>(src_h) / dst_h;
  const float scale_w = static_cast<float>(src_w) / dst_w;
  for (int dy = 0; dy < dst_h; ++dy) {
    const int sy = std::min(static_cast<int>(dy * scale_h), src_h - 1);
    for (int dx = 0; dx < dst_w; ++dx) {
      const int sx = std::min(static_cast<int>(dx * scale_w), src_w - 1);
      dst[dy * dst_w + dx] = src[sy * src_w + sx];
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Timer helper
// ═══════════════════════════════════════════════════════════════════════

class Timer {
 public:
  using Clock = std::chrono::high_resolution_clock;
  using Dur = std::chrono::microseconds;
  void start() { t0_ = Clock::now(); }
  double elapsed_ms() const {
    return static_cast<double>(
               std::chrono::duration_cast<Dur>(Clock::now() - t0_).count()) /
           1000.0;
  }

 private:
  Clock::time_point t0_;
};

// ═══════════════════════════════════════════════════════════════════════
// Inference engine
// ═══════════════════════════════════════════════════════════════════════

struct SkyWaterInference {
  Ort::Env env;
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::string input_name;
  std::string output_name;
  bool output_is_fp16 = false;

  SkyWaterInference(const std::string& model_path,
                    const std::string& provider = "cpu") {
    Ort::SessionOptions opts;

    if (provider == "cuda") {
#ifdef __APPLE__
      std::cerr << "WARNING: CUDA not available on macOS. "
                << "Falling back to CPU." << std::endl;
#else
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = 0;
      try {
        opts.AppendExecutionProvider_CUDA(cuda_opts);
      } catch (const Ort::Exception& e) {
        std::cerr << "WARNING: CUDA unavailable (" << e.what()
                  << "). Falling back to CPU." << std::endl;
      }
#endif
    }
    // Note: DML requires a separate DirectML-enabled ORT package.
    // "cpu" → default, no EP to append.

    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    session = std::make_unique<Ort::Session>(env, wpath.c_str(), opts);
#else
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
#endif

    // Query I/O names
    Ort::AllocatedStringPtr in_name = session->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr out_name = session->GetOutputNameAllocated(0, allocator);
    input_name = in_name.get();
    output_name = out_name.get();

    // Check output type
    Ort::TypeInfo out_type = session->GetOutputTypeInfo(0);
    auto tensor_info = out_type.GetTensorTypeAndShapeInfo();
    output_is_fp16 = (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

    std::cout << "Model: " << model_path << "\n"
              << "  Input:  " << input_name << " [1, 3, " << INPUT_H << ", " << INPUT_W << "]\n"
              << "  Output: " << output_name << " [1, 4, H, W] "
              << (output_is_fp16 ? "float16" : "float32") << "\n"
              << "  Provider: " << provider << "\n"
              << std::endl;
  }

  // Run inference on preprocessed float32 NCHW input.
  // Writes (1, 4, out_h, out_w) float32 logits into `out`.
  void run(const float* input_data,
           const std::array<int64_t, 4>& input_shape,
           int64_t& out_h, int64_t& out_w,
           std::vector<float>& out) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    size_t input_count = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data),
        input_count,
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};

    Ort::RunOptions run_opts;
    auto outputs = session->Run(run_opts, input_names, &input_tensor, 1,
                                output_names, 1);

    auto& out_val = outputs[0];
    auto out_info = out_val.GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();

    out_h = (out_shape.size() >= 3) ? out_shape[2] : INPUT_H;
    out_w = (out_shape.size() >= 4) ? out_shape[3] : INPUT_W;
    size_t out_count = out_shape[0] * out_shape[1] * out_h * out_w;

    out.resize(out_count);

    if (output_is_fp16) {
      const auto* fp16_data = out_val.GetTensorData<Ort::Float16_t>();
      for (size_t i = 0; i < out_count; ++i)
        out[i] = kHalfToFloatLUT[fp16_data[i].val];
    } else {
      const auto* fp32_data = out_val.GetTensorData<float>();
      std::copy_n(fp32_data, out_count, out.begin());
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════
// Preprocessing
// ═══════════════════════════════════════════════════════════════════════

// Load image → RGB uint8 (HWC, interleaved), returns {data, w, h, channels}
struct ImageRGB {
  std::vector<uint8_t> data;
  int w = 0, h = 0, channels = 0;
};

static ImageRGB load_image_rgb(const char* path) {
  ImageRGB img;
  unsigned char* raw = stbi_load(path, &img.w, &img.h, &img.channels, 3);
  if (!raw) {
    std::cerr << "ERROR: Cannot load image: " << path << "\n"
              << "  stbi reason: " << stbi_failure_reason() << std::endl;
    return {};
  }
  img.data.assign(raw, raw + img.w * img.h * 3);
  stbi_image_free(raw);
  img.channels = 3;
  return img;
}

// Preprocess: uint8 HWC → float CHW, resize, normalise.
// Fuses uint8→float + HWC→CHW + normalize into minimal passes.
// Returns (1, 3, INPUT_H, INPUT_W) float32 array.
static std::vector<float> preprocess(const ImageRGB& img) {
  constexpr float inv_255 = 1.0f / 255.0f;

  // Pass 1: uint8 HWC → float CHW (fuse type conversion + layout transpose)
  std::vector<float> src_chw(img.h * img.w * 3);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < img.h; ++y)
      for (int x = 0; x < img.w; ++x)
        src_chw[c * img.h * img.w + y * img.w + x] =
            static_cast<float>(img.data[(y * img.w + x) * 3 + c]);

  // Pass 2: bilinear resize CHW → CHW
  std::vector<float> dst_chw(INPUT_H * INPUT_W * 3);
  bilinear_resize(src_chw.data(), img.h, img.w,
                  dst_chw.data(), INPUT_H, INPUT_W, 3);

  // Pass 3: normalise in-place CHW → NCHW
  //   norm = (pixel / 255 - mean) / std  =  pixel * scale + bias
  std::vector<float> nchw(1 * 3 * INPUT_H * INPUT_W);
  for (int c = 0; c < 3; ++c) {
    const float scale = inv_255 / IMAGENET_STD[c];
    const float bias = -IMAGENET_MEAN[c] / IMAGENET_STD[c];
    const int off = c * INPUT_H * INPUT_W;
    for (int i = 0; i < INPUT_H * INPUT_W; ++i)
      nchw[off + i] = dst_chw[off + i] * scale + bias;
  }

  return nchw;
}

// ═══════════════════════════════════════════════════════════════════════
// Postprocessing
// ═══════════════════════════════════════════════════════════════════════

// Unrolled 4-way argmax (NUM_CLASSES is compile-time constant)
static std::vector<uint8_t> argmax_mask(const std::vector<float>& logits,
                                        int64_t h, int64_t w, int classes) {
  (void)classes;  // always NUM_CLASSES
  std::vector<uint8_t> mask(h * w);
  const int64_t stride = h * w;
  for (int64_t i = 0; i < stride; ++i) {
    float v0 = logits[0 * stride + i];
    float v1 = logits[1 * stride + i];
    float v2 = logits[2 * stride + i];
    float v3 = logits[3 * stride + i];

    uint8_t best = 0;
    float best_val = v0;
    if (v1 > best_val) { best_val = v1; best = 1; }
    if (v2 > best_val) { best_val = v2; best = 2; }
    if (v3 > best_val) { best_val = v3; best = 3; }
    mask[i] = best;
  }
  return mask;
}

// Colorise class-index mask to RGB uint8 (HWC)
static std::vector<uint8_t> colorize_mask(const std::vector<uint8_t>& mask,
                                          int h, int w) {
  std::vector<uint8_t> rgb(h * w * 3);
  for (int i = 0; i < h * w; ++i) {
    uint8_t cls = mask[i];
    if (cls >= NUM_CLASSES) cls = 0;
    rgb[i * 3 + 0] = CLASS_COLORS[cls][0];
    rgb[i * 3 + 1] = CLASS_COLORS[cls][1];
    rgb[i * 3 + 2] = CLASS_COLORS[cls][2];
  }
  return rgb;
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

// Overlay mask on original image with alpha blending
static std::vector<uint8_t> create_overlay(const uint8_t* original_rgb,
                                           const std::vector<uint8_t>& mask,
                                           int h, int w, float alpha = 0.45f) {
  std::vector<uint8_t> overlay(h * w * 3);
  for (int i = 0; i < h * w; ++i) {
    uint8_t cls = mask[i];
    if (cls >= NUM_CLASSES) cls = 0;
    if (cls == 0) {
      // Background: keep original pixel
      overlay[i * 3 + 0] = original_rgb[i * 3 + 0];
      overlay[i * 3 + 1] = original_rgb[i * 3 + 1];
      overlay[i * 3 + 2] = original_rgb[i * 3 + 2];
    } else {
      // Foreground: blend with class colour
      overlay[i * 3 + 0] = static_cast<uint8_t>(
          original_rgb[i * 3 + 0] * (1.0f - alpha) +
          CLASS_COLORS[cls][0] * alpha);
      overlay[i * 3 + 1] = static_cast<uint8_t>(
          original_rgb[i * 3 + 1] * (1.0f - alpha) +
          CLASS_COLORS[cls][1] * alpha);
      overlay[i * 3 + 2] = static_cast<uint8_t>(
          original_rgb[i * 3 + 2] * (1.0f - alpha) +
          CLASS_COLORS[cls][2] * alpha);
    }
  }
  return overlay;
}

static void print_usage() {
  std::cout
      << "Usage:\n"
      << "  skywater_infer.exe <model.onnx> <input_image> <output.png>\n"
      << "                     [cpu|cuda] [--iters N] [--overlay <path>]\n"
      << "\n"
      << "  model.onnx    : Path to FP16 (or FP32) ONNX model\n"
      << "  input_image   : JPEG / PNG image file\n"
      << "  output.png    : Where to save colourised segmentation mask\n"
      << "  cpu|cuda      : Execution provider (default: cpu)\n"
      << "  --iters N     : Number of inference runs for timing (default: 50)\n"
      << "  --overlay <p> : Save alpha-blended overlay to <p>\n"
      << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    print_usage();
    return 1;
  }

  std::string model_path = argv[1];
  std::string input_path = argv[2];
  std::string output_path = argv[3];
  std::string provider = "cpu";
  std::string overlay_path;
  int num_iters = 50;

  // Parse optional args
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "cpu" || arg == "cuda") {
      provider = arg;
    } else if (arg == "--iters" && i + 1 < argc) {
      num_iters = std::stoi(argv[++i]);
    } else if (arg == "--overlay" && i + 1 < argc) {
      overlay_path = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  // Validate files
  if (!std::filesystem::exists(model_path)) {
    std::cerr << "ERROR: Model not found: " << model_path << "\n";
    return 1;
  }
  if (!std::filesystem::exists(input_path)) {
    std::cerr << "ERROR: Input image not found: " << input_path << "\n";
    return 1;
  }

  // ---------- Load model ----------
  Timer timer;
  SkyWaterInference infer(model_path, provider);

  // ---------- Load & preprocess image ----------
  std::cout << "Loading image: " << input_path << std::endl;
  ImageRGB img = load_image_rgb(input_path.c_str());
  if (img.data.empty()) return 1;
  std::cout << "  Original size: " << img.w << "×" << img.h
            << " (" << img.channels << " channels)" << std::endl;

  timer.start();
  auto nchw_input = preprocess(img);
  std::cout << "  Preprocess time: " << timer.elapsed_ms() << " ms" << std::endl;

  // ---------- Warm-up ----------
  std::array<int64_t, 4> input_shape{1, 3, INPUT_H, INPUT_W};
  int64_t out_h = 0, out_w = 0;
  std::vector<float> result;  // pre-allocated — reused across all iters
  result.reserve(1 * NUM_CLASSES * INPUT_H * INPUT_W);

  std::cout << "Warm-up run..." << std::endl;
  infer.run(nchw_input.data(), input_shape, out_h, out_w, result);

  // ---------- Timed inference ----------
  std::cout << "Running " << num_iters << " inferences..." << std::endl;
  std::vector<double> times;
  times.reserve(num_iters);

  for (int i = 0; i < num_iters; ++i) {
    timer.start();
    infer.run(nchw_input.data(), input_shape, out_h, out_w, result);
    double ms = timer.elapsed_ms();
    times.push_back(ms);

    // Keep the last result for postprocessing
    if (i == num_iters - 1) {
      // ---------- Postprocess ----------
      timer.start();
      auto mask_384 = argmax_mask(result, out_h, out_w, NUM_CLASSES);

      // Resize mask to original size
      std::vector<uint8_t> mask_orig(img.h * img.w);
      nearest_resize_uint8(mask_384.data(), out_h, out_w,
                           mask_orig.data(), img.h, img.w);

      // Colorise and save
      auto rgb = colorize_mask(mask_orig, img.h, img.w);
      int write_ok = stbi_write_png(output_path.c_str(), img.w, img.h, 3,
                                    rgb.data(), img.w * 3);

      // Save overlay if requested
      if (!overlay_path.empty()) {
        auto overlay = create_overlay(img.data.data(), mask_orig, img.h, img.w);
        int ov_ok = stbi_write_png(overlay_path.c_str(), img.w, img.h, 3,
                                   overlay.data(), img.w * 3);
        if (ov_ok == 0) {
          std::cerr << "WARNING: Failed to write overlay: " << overlay_path
                    << "\n  stbi reason: " << stbi_failure_reason() << std::endl;
        } else {
          std::cout << "  Overlay saved: " << overlay_path << std::endl;
        }
      }
      double post_ms = timer.elapsed_ms();

      if (write_ok == 0) {
        std::cerr << "ERROR: Failed to write output: " << output_path
                  << "\n  stbi reason: " << stbi_failure_reason() << std::endl;
        return 1;
      }
      std::cout << "  Postprocess time: " << post_ms << " ms" << std::endl;

      // Class statistics
      int counts[NUM_CLASSES] = {};
      for (size_t j = 0; j < mask_orig.size(); ++j)
        ++counts[mask_orig[j] >= NUM_CLASSES ? 0 : mask_orig[j]];
      float total = static_cast<float>(mask_orig.size());
      std::cout << "  Saved mask: " << output_path
                << " (" << img.w << "×" << img.h << ")\n"
                << "  bg: " << counts[0] / total * 100.0f << "%"
                << "  sky: " << counts[1] / total * 100.0f << "%"
                << "  water: " << counts[2] / total * 100.0f << "%"
                << "  person: " << counts[3] / total * 100.0f << "%"
                << std::endl;
    }
  }

  // ---------- Timing report ----------
  std::sort(times.begin(), times.end());
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double avg = sum / times.size();
  double p50 = times[times.size() / 2];
  double p95 = times[static_cast<size_t>(times.size() * 0.95)];
  double min_t = times.front();
  double max_t = times.back();

  // Compute stddev
  double sq_sum = 0;
  for (double t : times) sq_sum += (t - avg) * (t - avg);
  double stddev = std::sqrt(sq_sum / times.size());

  std::cout << "\n========== Timing Results ==========\n"
            << "  Iters:   " << num_iters << "\n"
            << "  Min:     " << min_t << " ms\n"
            << "  Avg:     " << avg << " ms\n"
            << "  Median:  " << p50 << " ms\n"
            << "  P95:     " << p95 << " ms\n"
            << "  Max:     " << max_t << " ms\n"
            << "  StdDev:  " << stddev << " ms\n"
            << "=====================================" << std::endl;

  return 0;
}
