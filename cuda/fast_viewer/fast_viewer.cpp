#include "../raytracer.h"

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif

cudaError_t upload_rgb_to_rgba_array(cudaArray_t dst, const float *rgb, int width, int height, cudaStream_t stream);

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Args {
    fs::path model_path;
    int iteration = -1;
    bool benchmark = false;
    double benchmark_seconds = 3.0;
    std::string benchmark_camera_split;
    fs::path dump_frame_path;
    fs::path dump_tensor_path;
};

struct ViewerConfig {
    int64_t ppll_forward_size = 300000000;
    bool sh = true;
    int64_t sh_max_degree = 3;
    float exp_power = 2.0f;
    float alpha_threshold = 0.01f;
    float t_threshold = 0.03f;
    std::array<float, 3> bg_color = {0.0f, 0.0f, 0.0f};
};

struct CameraState {
    int width = 1280;
    int height = 720;
    float fov_y = 0.7f;
    std::array<float, 3> origin = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> right = {1.0f, 0.0f, 0.0f};
    std::array<float, 3> up = {0.0f, -1.0f, 0.0f};
    std::array<float, 3> forward = {0.0f, 0.0f, 1.0f};
    std::array<float, 3> smoothed_origin_motion = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> smoothed_rotation_motion = {0.0f, 0.0f, 0.0f};
};

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
};

struct LoadedCameras {
    std::vector<CameraState> all;
    std::vector<CameraState> train;
    std::vector<CameraState> test;
};

static std::string read_text(const fs::path &path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), {});
}

static double json_number(const std::string &json, const std::string &key, double fallback) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)");
    std::smatch m;
    return std::regex_search(json, m, re) ? std::stod(m[1].str()) : fallback;
}

static bool json_bool(const std::string &json, const std::string &key, bool fallback) {
    std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch m;
    return std::regex_search(json, m, re) ? (m[1].str() == "true") : fallback;
}

static std::vector<double> json_array_numbers(const std::string &json, const std::string &key) {
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return {};
    }
    size_t begin = json.find('[', key_pos);
    if (begin == std::string::npos) {
        return {};
    }
    int depth = 0;
    size_t end = begin;
    for (; end < json.size(); ++end) {
        if (json[end] == '[') {
            ++depth;
        } else if (json[end] == ']') {
            --depth;
            if (depth == 0) {
                ++end;
                break;
            }
        }
    }
    std::string body = json.substr(begin, end - begin);
    std::regex number("-?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?");
    std::vector<double> values;
    for (auto it = std::sregex_iterator(body.begin(), body.end(), number); it != std::sregex_iterator(); ++it) {
        values.push_back(std::stod((*it)[0].str()));
    }
    return values;
}

static std::vector<std::string> json_top_level_objects(const std::string &json) {
    std::vector<std::string> objects;
    int bracket_depth = 0;
    int brace_depth = 0;
    size_t object_begin = std::string::npos;
    bool in_string = false;
    bool escape = false;
    for (size_t i = 0; i < json.size(); ++i) {
        char c = json[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '[') {
            ++bracket_depth;
        } else if (c == ']') {
            --bracket_depth;
        } else if (c == '{') {
            if (bracket_depth == 1 && brace_depth == 0) {
                object_begin = i;
            }
            ++brace_depth;
        } else if (c == '}') {
            --brace_depth;
            if (bracket_depth == 1 && brace_depth == 0 && object_begin != std::string::npos) {
                objects.push_back(json.substr(object_begin, i - object_begin + 1));
                object_begin = std::string::npos;
            }
        }
    }
    return objects;
}

static ViewerConfig load_config(const fs::path &model_path) {
    ViewerConfig cfg;
    std::string json = read_text(model_path / "config.json");
    cfg.ppll_forward_size = static_cast<int64_t>(json_number(json, "ppll_forward_size", cfg.ppll_forward_size));
    cfg.sh = json_bool(json, "sh", cfg.sh);
    cfg.sh_max_degree = cfg.sh ? static_cast<int64_t>(json_number(json, "sh_max_degree", cfg.sh_max_degree)) : 0;
    cfg.exp_power = static_cast<float>(json_number(json, "exp_power", cfg.exp_power));
    cfg.alpha_threshold = static_cast<float>(json_number(json, "alpha_threshold", cfg.alpha_threshold));
    cfg.t_threshold = static_cast<float>(json_number(json, "t_threshold", cfg.t_threshold));
    std::vector<double> bg = json_array_numbers(json, "bg_color");
    if (bg.size() >= 3) {
        cfg.bg_color = {static_cast<float>(bg[0]), static_cast<float>(bg[1]), static_cast<float>(bg[2])};
    }
    return cfg;
}

static CameraState parse_camera_object(const std::string &json) {
    CameraState cam;
    std::vector<double> r = json_array_numbers(json, "R");
    std::vector<double> o = json_array_numbers(json, "origin");
    if (r.size() >= 9) {
        cam.right = {static_cast<float>(r[0]), static_cast<float>(r[3]), static_cast<float>(r[6])};
        cam.up = {-static_cast<float>(r[1]), -static_cast<float>(r[4]), -static_cast<float>(r[7])};
        cam.forward = {static_cast<float>(r[2]), static_cast<float>(r[5]), static_cast<float>(r[8])};
    }
    if (o.size() >= 3) {
        cam.origin = {static_cast<float>(o[0]), static_cast<float>(o[1]), static_cast<float>(o[2])};
    }
    cam.fov_y = static_cast<float>(json_number(json, "fov_y", cam.fov_y));
    cam.width = static_cast<int>(json_number(json, "image_width", cam.width));
    cam.height = static_cast<int>(json_number(json, "image_height", cam.height));
    return cam;
}

static LoadedCameras load_cameras(const fs::path &model_path) {
    LoadedCameras result;
    std::string json = read_text(model_path / "cameras.json");
    for (const std::string &object : json_top_level_objects(json)) {
        CameraState cam = parse_camera_object(object);
        result.all.push_back(cam);
        if (json_bool(object, "is_test", false)) {
            result.test.push_back(cam);
        } else {
            result.train.push_back(cam);
        }
    }
    if (result.all.empty()) {
        result.all.push_back(parse_camera_object(json));
    }
    return result;
}

static TensorInfo parse_tensor_info(const std::string &header, const std::string &key) {
    size_t key_pos = header.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        throw std::runtime_error("safetensors missing key: " + key);
    }
    size_t begin = header.find('{', key_pos);
    int depth = 0;
    size_t end = begin;
    for (; end < header.size(); ++end) {
        if (header[end] == '{') {
            ++depth;
        } else if (header[end] == '}') {
            --depth;
            if (depth == 0) {
                ++end;
                break;
            }
        }
    }
    std::string obj = header.substr(begin, end - begin);
    TensorInfo info;
    std::smatch m;
    std::regex dtype_re("\"dtype\"\\s*:\\s*\"([A-Z0-9]+)\"");
    std::regex offsets_re("\"data_offsets\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    if (!std::regex_search(obj, m, dtype_re)) {
        throw std::runtime_error("safetensors dtype missing for " + key);
    }
    info.dtype = m[1].str();
    if (!std::regex_search(obj, m, offsets_re)) {
        throw std::runtime_error("safetensors offsets missing for " + key);
    }
    info.begin = std::stoull(m[1].str());
    info.end = std::stoull(m[2].str());
    size_t shape_pos = obj.find("\"shape\"");
    size_t shape_begin = obj.find('[', shape_pos);
    size_t shape_end = obj.find(']', shape_begin);
    std::string shape = obj.substr(shape_begin, shape_end - shape_begin + 1);
    std::regex integer("-?[0-9]+");
    for (auto it = std::sregex_iterator(shape.begin(), shape.end(), integer); it != std::sregex_iterator(); ++it) {
        info.shape.push_back(std::stoll((*it)[0].str()));
    }
    return info;
}

static torch::Dtype tensor_dtype(const std::string &dtype) {
    if (dtype == "F32") {
        return torch::kFloat32;
    }
    if (dtype == "I32") {
        return torch::kInt32;
    }
    if (dtype == "I64") {
        return torch::kInt64;
    }
    throw std::runtime_error("unsupported safetensors dtype: " + dtype);
}

static torch::Tensor load_tensor(std::ifstream &file, uint64_t data_start, const std::string &header,
                                 const std::string &key) {
    TensorInfo info = parse_tensor_info(header, key);
    torch::Tensor cpu = torch::empty(info.shape, torch::TensorOptions().dtype(tensor_dtype(info.dtype)).device(torch::kCPU));
    uint64_t bytes = info.end - info.begin;
    if (bytes != static_cast<uint64_t>(cpu.nbytes())) {
        throw std::runtime_error("safetensors byte count mismatch for " + key);
    }
    file.seekg(static_cast<std::streamoff>(data_start + info.begin));
    file.read(reinterpret_cast<char *>(cpu.data_ptr()), static_cast<std::streamsize>(bytes));
    if (!file) {
        throw std::runtime_error("failed reading tensor " + key);
    }
    return cpu.to(torch::kCUDA, true);
}

static std::string read_safetensors_header(std::ifstream &file, uint64_t &data_start) {
    uint64_t header_len = 0;
    file.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
    std::string header(header_len, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header_len));
    data_start = sizeof(header_len) + header_len;
    return header;
}

static fs::path find_checkpoint(const fs::path &model_path, int iteration) {
    if (model_path.extension() == ".safetensors") {
        return model_path;
    }
    if (iteration >= 0) {
        std::ostringstream name;
        name << "gaussians_" << std::setw(5) << std::setfill('0') << iteration << ".safetensors";
        return model_path / name.str();
    }
    int best = -1;
    fs::path best_path;
    for (const auto &entry : fs::directory_iterator(model_path)) {
        std::string name = entry.path().filename().string();
        std::smatch m;
        if (std::regex_match(name, m, std::regex("gaussians_([0-9]+)\\.safetensors"))) {
            int value = std::stoi(m[1].str());
            if (value > best) {
                best = value;
                best_path = entry.path();
            }
        }
    }
    if (best_path.empty()) {
        throw std::runtime_error("no gaussians_*.safetensors found in " + model_path.string());
    }
    return best_path;
}

static void load_gaussians(Raytracer &rt, const fs::path &checkpoint, bool sh_enabled) {
    std::ifstream file(checkpoint, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + checkpoint.string());
    }
    uint64_t data_start = 0;
    std::string header = read_safetensors_header(file, data_start);
    torch::NoGradGuard no_grad;
    auto g = rt.gaussian_data;
    g->mean.copy_(load_tensor(file, data_start, header, "mean"));
    g->rotation.copy_(load_tensor(file, data_start, header, "rotation"));
    g->scale.copy_(load_tensor(file, data_start, header, "scale"));
    g->opacity.copy_(load_tensor(file, data_start, header, "opacity"));
    g->channels.copy_(load_tensor(file, data_start, header, "channels"));
    g->sh_coeffs_dc.copy_(load_tensor(file, data_start, header, "sh_coeffs_dc"));
    if (sh_enabled) {
        g->sh_coeffs_rest.copy_(load_tensor(file, data_start, header, "sh_coeffs_rest"));
    }
    g->current_sh_degree.copy_(load_tensor(file, data_start, header, "current_sh_degree"));
}

static void configure_raytracer(Raytracer &rt, const ViewerConfig &cfg) {
    torch::NoGradGuard no_grad;
    auto c = rt.config_data;
    c->alpha_threshold.fill_(cfg.alpha_threshold);
    c->t_threshold.fill_(cfg.t_threshold);
    c->exp_power.fill_(cfg.exp_power);
    c->render_depth.fill_(false);
    c->rays_from_python.fill_(false);
    c->background_channels.copy_(torch::tensor({cfg.bg_color[0], cfg.bg_color[1], cfg.bg_color[2]},
                                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)));
    c->enable_sh.fill_(cfg.sh);
    c->update_channels.fill_(!cfg.sh);
    rt.meta_data->grads_enabled.fill_(false);
    rt.meta_data->run_forward_pass.fill_(true);
    rt.meta_data->run_backward_pass.fill_(false);
}

static void upload_camera(Raytracer &rt, const CameraState &cam) {
    torch::NoGradGuard no_grad;
    rt.camera_data->znear.fill_(0.0f);
    rt.camera_data->zfar.fill_(99999.9f);
    rt.camera_data->vertical_fov_radians.fill_(cam.fov_y);
    std::array<float, 9> c2w = {
        cam.right[0], cam.up[0], -cam.forward[0],
        cam.right[1], cam.up[1], -cam.forward[1],
        cam.right[2], cam.up[2], -cam.forward[2],
    };
    torch::Tensor origin = torch::from_blob(const_cast<float *>(cam.origin.data()), {3},
                                            torch::TensorOptions().dtype(torch::kFloat32)).clone().to(torch::kCUDA);
    torch::Tensor rot = torch::from_blob(c2w.data(), {3, 3},
                                         torch::TensorOptions().dtype(torch::kFloat32)).clone().to(torch::kCUDA);
    rt.camera_data->set_pose(origin, rot);
}

static float dot3(const std::array<float, 3> &a, const std::array<float, 3> &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static std::array<float, 3> cross3(const std::array<float, 3> &a, const std::array<float, 3> &b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

static float length3(const std::array<float, 3> &v) {
    return std::sqrt(std::max(dot3(v, v), 1e-20f));
}

static void normalize3(std::array<float, 3> &v) {
    float inv_len = 1.0f / length3(v);
    v[0] *= inv_len;
    v[1] *= inv_len;
    v[2] *= inv_len;
}

static void add_scaled(std::array<float, 3> &v, const std::array<float, 3> &d, float s) {
    v[0] += d[0] * s;
    v[1] += d[1] * s;
    v[2] += d[2] * s;
}

static std::array<float, 3> rotate_vec(const std::array<float, 3> &vec, std::array<float, 3> axis, float angle) {
    normalize3(axis);
    float c = std::cos(angle);
    float s = std::sin(angle);
    float d = dot3(axis, vec);
    std::array<float, 3> cross = cross3(axis, vec);
    return {
        c * vec[0] + s * cross[0] + (1.0f - c) * d * axis[0],
        c * vec[1] + s * cross[1] + (1.0f - c) * d * axis[1],
        c * vec[2] + s * cross[2] + (1.0f - c) * d * axis[2],
    };
}

static bool apply_rotation(CameraState &cam, float angle_forward, float angle_right, float angle_up) {
    bool changed = false;
    if (std::abs(angle_forward) > 1e-7f) {
        cam.up = rotate_vec(cam.up, cam.forward, angle_forward);
        cam.right = rotate_vec(cam.right, cam.forward, angle_forward);
        changed = true;
    }
    if (std::abs(angle_right) > 1e-7f) {
        cam.forward = rotate_vec(cam.forward, cam.right, angle_right);
        cam.up = rotate_vec(cam.up, cam.right, angle_right);
        changed = true;
    }
    if (std::abs(angle_up) > 1e-7f) {
        cam.forward = rotate_vec(cam.forward, cam.up, angle_up);
        cam.right = rotate_vec(cam.right, cam.up, angle_up);
        changed = true;
    }
    if (!changed) {
        return false;
    }

    normalize3(cam.forward);
    cam.right = cross3(cam.forward, cam.up);
    normalize3(cam.right);
    cam.up = cross3(cam.right, cam.forward);
    normalize3(cam.up);
    return true;
}

static bool almost_nonzero(const std::array<float, 3> &v) {
    return dot3(v, v) > 1e-14f;
}

static bool key_down(const char keys[32], KeyCode code) {
    return (keys[code / 8] & (1 << (code % 8))) != 0;
}

static void put_u32_be(std::vector<uint8_t> &out, uint32_t value) {
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>(value & 0xff));
}

static uint32_t crc32_bytes(const uint8_t *data, size_t size) {
    static uint32_t table[256] = {};
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        ready = true;
    }
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        c = table[(c ^ data[i]) & 0xff] ^ (c >> 8);
    }
    return c ^ 0xffffffffu;
}

static uint32_t adler32_bytes(const uint8_t *data, size_t size) {
    uint32_t a = 1;
    uint32_t b = 0;
    for (size_t i = 0; i < size; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

static void append_png_chunk(std::vector<uint8_t> &png, const char type[4], const std::vector<uint8_t> &data) {
    put_u32_be(png, static_cast<uint32_t>(data.size()));
    size_t chunk_begin = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), data.begin(), data.end());
    put_u32_be(png, crc32_bytes(png.data() + chunk_begin, png.size() - chunk_begin));
}

static void write_png_rgb8(const fs::path &path, const std::vector<uint8_t> &rgb, int width, int height) {
    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};

    std::vector<uint8_t> ihdr;
    put_u32_be(ihdr, static_cast<uint32_t>(width));
    put_u32_be(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8);
    ihdr.push_back(2);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    append_png_chunk(png, "IHDR", ihdr);

    std::vector<uint8_t> filtered;
    filtered.reserve(static_cast<size_t>(height) * (static_cast<size_t>(width) * 3 + 1));
    for (int y = 0; y < height; ++y) {
        filtered.push_back(0);
        const uint8_t *row = rgb.data() + static_cast<size_t>(y) * width * 3;
        filtered.insert(filtered.end(), row, row + static_cast<size_t>(width) * 3);
    }

    std::vector<uint8_t> zlib;
    zlib.reserve(filtered.size() + filtered.size() / 65535 * 5 + 16);
    zlib.push_back(0x78);
    zlib.push_back(0x01);
    size_t pos = 0;
    while (pos < filtered.size()) {
        uint16_t block = static_cast<uint16_t>(std::min<size_t>(65535, filtered.size() - pos));
        bool final = pos + block == filtered.size();
        zlib.push_back(final ? 1 : 0);
        zlib.push_back(static_cast<uint8_t>(block & 0xff));
        zlib.push_back(static_cast<uint8_t>((block >> 8) & 0xff));
        uint16_t nlen = static_cast<uint16_t>(~block);
        zlib.push_back(static_cast<uint8_t>(nlen & 0xff));
        zlib.push_back(static_cast<uint8_t>((nlen >> 8) & 0xff));
        zlib.insert(zlib.end(), filtered.begin() + static_cast<std::ptrdiff_t>(pos),
                    filtered.begin() + static_cast<std::ptrdiff_t>(pos + block));
        pos += block;
    }
    put_u32_be(zlib, adler32_bytes(filtered.data(), filtered.size()));
    append_png_chunk(png, "IDAT", zlib);

    append_png_chunk(png, "IEND", {});
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open dump path " + path.string());
    }
    file.write(reinterpret_cast<const char *>(png.data()), static_cast<std::streamsize>(png.size()));
}

static void dump_front_buffer_png(const fs::path &path, int width, int height) {
    std::vector<uint8_t> bottom_up(static_cast<size_t>(width) * height * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, bottom_up.data());
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        throw std::runtime_error("glReadPixels failed with GL error " + std::to_string(err));
    }

    std::vector<uint8_t> top_down(bottom_up.size());
    const size_t row_bytes = static_cast<size_t>(width) * 3;
    for (int y = 0; y < height; ++y) {
        std::memcpy(top_down.data() + static_cast<size_t>(y) * row_bytes,
                    bottom_up.data() + static_cast<size_t>(height - 1 - y) * row_bytes, row_bytes);
    }
    write_png_rgb8(path, top_down, width, height);
}

static void dump_tensor_png(const fs::path &path, const torch::Tensor &tensor, int width, int height) {
    torch::Tensor cpu = tensor.detach().clamp(0.0, 1.0).mul(255.0).to(torch::kUInt8).to(torch::kCPU).contiguous();
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
    std::memcpy(rgb.data(), cpu.data_ptr(), rgb.size());
    write_png_rgb8(path, rgb, width, height);
}

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model-path") {
            args.model_path = need_value(a);
        } else if (a == "-t" || a == "--iteration") {
            args.iteration = std::stoi(need_value(a));
        } else if (a == "--benchmark") {
            args.benchmark = true;
        } else if (a == "--benchmark-seconds") {
            args.benchmark_seconds = std::stod(need_value(a));
            args.benchmark = true;
        } else if (a == "--benchmark-test-cameras") {
            args.benchmark_camera_split = "test";
        } else if (a == "--benchmark-cameras") {
            args.benchmark_camera_split = need_value(a);
            if (args.benchmark_camera_split != "train" && args.benchmark_camera_split != "test") {
                throw std::runtime_error("--benchmark-cameras must be 'train' or 'test'");
            }
        } else if (a == "--dump-frame") {
            args.dump_frame_path = need_value(a);
        } else if (a == "--dump-tensor") {
            args.dump_tensor_path = need_value(a);
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: fast_viewer -m MODEL_PATH [--iteration N] [--benchmark]"
                         " [--benchmark-test-cameras|--benchmark-cameras train|test]"
                         " [--dump-frame PATH.png] [--dump-tensor PATH.png]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (args.model_path.empty()) {
        throw std::runtime_error("pass -m MODEL_PATH");
    }
    return args;
}

static void disable_vsync(Display *display, GLXDrawable drawable) {
    using SwapIntervalEXT = void (*)(Display *, GLXDrawable, int);
    using SwapIntervalMESA = int (*)(unsigned int);
    auto ext = reinterpret_cast<SwapIntervalEXT>(glXGetProcAddressARB(reinterpret_cast<const GLubyte *>("glXSwapIntervalEXT")));
    if (ext) {
        ext(display, drawable, 0);
        return;
    }
    auto mesa = reinterpret_cast<SwapIntervalMESA>(glXGetProcAddressARB(reinterpret_cast<const GLubyte *>("glXSwapIntervalMESA")));
    if (mesa) {
        mesa(0);
    }
}

static void draw_texture(GLuint texture) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();
}

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        if (!fs::exists(args.model_path) && args.model_path == fs::path("output/bicycle") &&
            fs::exists("output/pretrained/bicycle")) {
            args.model_path = "output/pretrained/bicycle";
        }

        ViewerConfig cfg = load_config(args.model_path);
        LoadedCameras loaded_cameras = load_cameras(args.model_path);
        CameraState cam = !loaded_cameras.test.empty() ? loaded_cameras.test.front() : loaded_cameras.all.front();
        fs::path checkpoint = find_checkpoint(args.model_path, args.iteration);
        TensorInfo mean_info;
        {
            std::ifstream file(checkpoint, std::ios::binary);
            uint64_t data_start = 0;
            std::string header = read_safetensors_header(file, data_start);
            mean_info = parse_tensor_info(header, "mean");
        }

        Display *display = XOpenDisplay(nullptr);
        if (!display) {
            throw std::runtime_error("failed to open X display");
        }
        int screen = DefaultScreen(display);
        int visual_attrs[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
        XVisualInfo *visual = glXChooseVisual(display, screen, visual_attrs);
        if (!visual) {
            throw std::runtime_error("failed to choose GLX visual");
        }
        Colormap cmap = XCreateColormap(display, RootWindow(display, screen), visual->visual, AllocNone);
        XSetWindowAttributes swa = {};
        swa.colormap = cmap;
        swa.event_mask = ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                         Button1MotionMask | StructureNotifyMask;
        Window window = XCreateWindow(display, RootWindow(display, screen), 0, 0, cam.width, cam.height, 0,
                                      visual->depth, InputOutput, visual->visual, CWColormap | CWEventMask, &swa);
        Atom wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display, window, &wm_delete, 1);
        XStoreName(display, window, "Gray Fast Viewer");
        XMapWindow(display, window);

        GLXContext gl_context = glXCreateContext(display, visual, nullptr, GL_TRUE);
        glXMakeCurrent(display, window, gl_context);
        disable_vsync(display, window);
        XWindowAttributes window_attrs = {};
        XGetWindowAttributes(display, window, &window_attrs);
        int window_width = std::max(1, window_attrs.width);
        int window_height = std::max(1, window_attrs.height);
        glViewport(0, 0, window_width, window_height);

        GLuint texture = 0;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, cam.width, cam.height, 0, GL_RGBA, GL_FLOAT, nullptr);

        cudaGraphicsResource *cuda_texture = nullptr;
        CUDA_CHECK(cudaGraphicsGLRegisterImage(&cuda_texture, texture, GL_TEXTURE_2D,
                                               cudaGraphicsRegisterFlagsWriteDiscard));

        std::cerr << "Loading " << mean_info.shape[0] << " gaussians from " << checkpoint << "\n";
        Raytracer rt(cam.width, cam.height, mean_info.shape[0], cfg.sh_max_degree, cfg.ppll_forward_size, 1,
                 false);
        configure_raytracer(rt, cfg);
        load_gaussians(rt, checkpoint, cfg.sh);
        rt.rebuild_bvh();
        upload_camera(rt, cam);

        auto render_to_texture = [&]() {
            {
                torch::NoGradGuard no_grad;
                rt.forward_pass_display();
            }

            cudaArray_t array = nullptr;
            CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_texture, 0));
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, cuda_texture, 0, 0));
            CUDA_CHECK(upload_rgb_to_rgba_array(array,
                                                static_cast<const float *>(rt.framebuffer_data->output_channels.data_ptr()),
                                                cam.width, cam.height, 0));
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_texture, 0));

            draw_texture(texture);
            glXSwapBuffers(display, window);
        };

        if (!args.benchmark_camera_split.empty()) {
            const std::vector<CameraState> &benchmark_cameras =
                args.benchmark_camera_split == "train" ? loaded_cameras.train : loaded_cameras.test;
            if (benchmark_cameras.empty()) {
                throw std::runtime_error("no " + args.benchmark_camera_split + " cameras found in cameras.json");
            }

            cam = benchmark_cameras.front();
            upload_camera(rt, cam);
            render_to_texture();
            XSync(display, False);

            std::cerr << "Warming " << benchmark_cameras.size() << " " << args.benchmark_camera_split
                      << " cameras\n";
            for (const CameraState &benchmark_cam : benchmark_cameras) {
                cam = benchmark_cam;
                upload_camera(rt, cam);
                torch::NoGradGuard no_grad;
                rt.forward_pass_display();
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::cerr << "Measuring " << args.benchmark_camera_split << " camera FPS\n";
            cudaEvent_t start_event = nullptr;
            cudaEvent_t end_event = nullptr;
            CUDA_CHECK(cudaEventCreate(&start_event));
            CUDA_CHECK(cudaEventCreate(&end_event));
            CUDA_CHECK(cudaEventRecord(start_event, 0));
            for (const CameraState &benchmark_cam : benchmark_cameras) {
                cam = benchmark_cam;
                upload_camera(rt, cam);
                torch::NoGradGuard no_grad;
                rt.forward_pass_display();
            }
            CUDA_CHECK(cudaEventRecord(end_event, 0));
            CUDA_CHECK(cudaEventSynchronize(end_event));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));
            CUDA_CHECK(cudaEventDestroy(start_event));
            CUDA_CHECK(cudaEventDestroy(end_event));

            double fps = static_cast<double>(benchmark_cameras.size()) / std::max(elapsed_ms / 1000.0, 1e-9);
            std::cout << std::fixed << std::setprecision(2) << fps << "\n";

            CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_texture));
            glDeleteTextures(1, &texture);
            glXMakeCurrent(display, None, nullptr);
            glXDestroyContext(display, gl_context);
            XDestroyWindow(display, window);
            XCloseDisplay(display);
            return 0;
        }

        const KeyCode key_w = XKeysymToKeycode(display, XK_w);
        const KeyCode key_a = XKeysymToKeycode(display, XK_a);
        const KeyCode key_s = XKeysymToKeycode(display, XK_s);
        const KeyCode key_d = XKeysymToKeycode(display, XK_d);
        const KeyCode key_q = XKeysymToKeycode(display, XK_q);
        const KeyCode key_e = XKeysymToKeycode(display, XK_e);
        const KeyCode key_u = XKeysymToKeycode(display, XK_u);
        const KeyCode key_o = XKeysymToKeycode(display, XK_o);
        const KeyCode key_i = XKeysymToKeycode(display, XK_i);
        const KeyCode key_k = XKeysymToKeycode(display, XK_k);
        const KeyCode key_j = XKeysymToKeycode(display, XK_j);
        const KeyCode key_l = XKeysymToKeycode(display, XK_l);
        const KeyCode key_escape = XKeysymToKeycode(display, XK_Escape);

        bool running = true;
        bool printed_benchmark = false;
        bool dumped_frame = false;
        bool left_mouse_down = false;
        int last_mouse_x = 0;
        int last_mouse_y = 0;
        int mouse_delta_x = 0;
        int mouse_delta_y = 0;
        int frames = 0;
        int title_frames = 0;
        auto last = Clock::now();
        auto start = last;
        auto title_start = last;

        while (running) {
            while (XPending(display) > 0) {
                XEvent event;
                XNextEvent(display, &event);
                if (event.type == ClientMessage && static_cast<Atom>(event.xclient.data.l[0]) == wm_delete) {
                    running = false;
                } else if (event.type == ConfigureNotify) {
                    window_width = std::max(1, event.xconfigure.width);
                    window_height = std::max(1, event.xconfigure.height);
                    glViewport(0, 0, window_width, window_height);
                } else if (event.type == ButtonPress && event.xbutton.button == Button1) {
                    left_mouse_down = true;
                    last_mouse_x = event.xbutton.x;
                    last_mouse_y = event.xbutton.y;
                } else if (event.type == ButtonRelease && event.xbutton.button == Button1) {
                    left_mouse_down = false;
                } else if (event.type == MotionNotify && left_mouse_down) {
                    mouse_delta_x += event.xmotion.x - last_mouse_x;
                    mouse_delta_y += event.xmotion.y - last_mouse_y;
                    last_mouse_x = event.xmotion.x;
                    last_mouse_y = event.xmotion.y;
                }
            }

            auto now = Clock::now();
            double dt = std::chrono::duration<double>(now - last).count();
            last = now;

            char keys[32];
            XQueryKeymap(display, keys);
            if (key_down(keys, key_escape)) {
                running = false;
            }
            if (!args.benchmark) {
                constexpr float smoothness = 0.4f;
                constexpr float speed = 1.0f;
                constexpr float rot_speed = 1.0f;
                constexpr float mouse_speed = 4.0f;
                constexpr float radians_per_pixel = 3.14159265358979323846f / 150.0f;

                bool camera_dirty = false;
                std::array<float, 3> origin_motion = {0.0f, 0.0f, 0.0f};
                std::array<float, 3> rotation_motion = {0.0f, 0.0f, 0.0f};

                if (key_down(keys, key_w)) {
                    add_scaled(origin_motion, cam.forward, 1.0f);
                }
                if (key_down(keys, key_a)) {
                    add_scaled(origin_motion, cam.right, -1.0f);
                }
                if (key_down(keys, key_q)) {
                    add_scaled(origin_motion, cam.up, -1.0f);
                }
                if (key_down(keys, key_s)) {
                    add_scaled(origin_motion, cam.forward, -1.0f);
                }
                if (key_down(keys, key_d)) {
                    add_scaled(origin_motion, cam.right, 1.0f);
                }
                if (key_down(keys, key_e)) {
                    add_scaled(origin_motion, cam.up, 1.0f);
                }

                if (key_down(keys, key_o)) {
                    rotation_motion[0] += 50.0f * radians_per_pixel;
                }
                if (key_down(keys, key_u)) {
                    rotation_motion[0] -= 50.0f * radians_per_pixel;
                }
                if (key_down(keys, key_i)) {
                    rotation_motion[1] += 50.0f * radians_per_pixel;
                }
                if (key_down(keys, key_k)) {
                    rotation_motion[1] -= 50.0f * radians_per_pixel;
                }
                if (key_down(keys, key_j)) {
                    rotation_motion[2] += 50.0f * radians_per_pixel;
                }
                if (key_down(keys, key_l)) {
                    rotation_motion[2] -= 50.0f * radians_per_pixel;
                }

                if (mouse_delta_x != 0 || mouse_delta_y != 0) {
                    float angle_right = -static_cast<float>(mouse_delta_y) * radians_per_pixel *
                                        static_cast<float>(dt) * mouse_speed;
                    float angle_up = -static_cast<float>(mouse_delta_x) * radians_per_pixel *
                                     static_cast<float>(dt) * mouse_speed;
                    camera_dirty = apply_rotation(cam, 0.0f, angle_right, angle_up) || camera_dirty;
                    mouse_delta_x = 0;
                    mouse_delta_y = 0;
                }

                float weight = 1.0f - std::exp(-static_cast<float>(dt) / (smoothness + 1e-6f));
                for (int idx = 0; idx < 3; ++idx) {
                    cam.smoothed_origin_motion[idx] =
                        cam.smoothed_origin_motion[idx] * (1.0f - weight) + origin_motion[idx] * weight;
                    cam.smoothed_rotation_motion[idx] =
                        cam.smoothed_rotation_motion[idx] * (1.0f - weight) + rotation_motion[idx] * weight;
                }

                if (almost_nonzero(cam.smoothed_origin_motion)) {
                    add_scaled(cam.origin, cam.smoothed_origin_motion, static_cast<float>(dt) * speed);
                    camera_dirty = true;
                }
                if (almost_nonzero(cam.smoothed_rotation_motion)) {
                    camera_dirty = apply_rotation(cam,
                                                  cam.smoothed_rotation_motion[0] * static_cast<float>(dt) * rot_speed,
                                                  cam.smoothed_rotation_motion[1] * static_cast<float>(dt) * rot_speed,
                                                  cam.smoothed_rotation_motion[2] * static_cast<float>(dt) * rot_speed) ||
                                   camera_dirty;
                }

                if (camera_dirty) {
                    upload_camera(rt, cam);
                }
            }

            {
                torch::NoGradGuard no_grad;
                rt.forward_pass_display();
            }

            if (!args.dump_tensor_path.empty() && !dumped_frame) {
                dump_tensor_png(args.dump_tensor_path, rt.framebuffer_data->output_channels, cam.width, cam.height);
                std::cerr << "Wrote CUDA framebuffer to " << args.dump_tensor_path << "\n";
            }

            cudaArray_t array = nullptr;
            CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_texture, 0));
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, cuda_texture, 0, 0));
            CUDA_CHECK(upload_rgb_to_rgba_array(array,
                                                static_cast<const float *>(rt.framebuffer_data->output_channels.data_ptr()),
                                                cam.width, cam.height, 0));
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_texture, 0));

            draw_texture(texture);
            glXSwapBuffers(display, window);
            ++frames;
            ++title_frames;

            if ((!args.dump_frame_path.empty() || !args.dump_tensor_path.empty()) && !dumped_frame) {
                glFinish();
                XSync(display, False);
                if (!args.dump_frame_path.empty()) {
                    dump_front_buffer_png(args.dump_frame_path, window_width, window_height);
                    std::cerr << "Wrote displayed frame to " << args.dump_frame_path << "\n";
                }
                dumped_frame = true;
                if (!args.benchmark) {
                    running = false;
                }
            }

            now = Clock::now();
            double title_elapsed = std::chrono::duration<double>(now - title_start).count();
            if (title_elapsed >= 0.25) {
                double fps = title_frames / title_elapsed;
                std::ostringstream title;
                title << "Gray Fast Viewer - " << std::fixed << std::setprecision(1) << fps << " FPS";
                XStoreName(display, window, title.str().c_str());
                title_frames = 0;
                title_start = now;
            }

            if (args.benchmark) {
                double elapsed = std::chrono::duration<double>(now - start).count();
                if (elapsed >= args.benchmark_seconds) {
                    double fps = frames / elapsed;
                    std::cout << std::fixed << std::setprecision(2) << fps << "\n";
                    printed_benchmark = true;
                    running = false;
                }
            }
        }

        if (args.benchmark && !printed_benchmark) {
            auto end = Clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << std::fixed << std::setprecision(2) << (frames / std::max(elapsed, 1e-9)) << "\n";
        }

        CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_texture));
        glDeleteTextures(1, &texture);
        glXMakeCurrent(display, None, nullptr);
        glXDestroyContext(display, gl_context);
        XDestroyWindow(display, window);
        XCloseDisplay(display);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "fast_viewer: " << e.what() << "\n";
        return 1;
    }
}
