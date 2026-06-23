// spec 041 T7: correctness test for ggml_fused_ffn.
//
// The fused FFN op computes
//     dst = down @ (silu(gate @ x) * (up @ x))           [hidden, n_tokens]
//
// We compute it three ways and compare:
//   1. CPU reference decomposition (mul_mat/silu/mul/mul_mat) on the CPU backend.
//   2. CPU fused op (ggml_fused_ffn -> CPU fallback subgraph).
//   3. Vulkan fused op (ggml_fused_ffn -> fused_ffn.comp), when a GPU supports it.
//
// Two comparisons per case:
//   (A) CPU-fused vs CPU-ref     -> validates the CPU fallback decomposes correctly
//                                  (expected ~0; isolates fallback graph bugs).
//   (B) Vulkan-fused vs CPU-fused -> validates the Vulkan kernel against the CPU op
//                                  (the real cross-backend kernel check; tol 5e-2).
//
// For batched shapes (n_tokens>1) no GPU advertises support (decode-only kernel),
// so (B) is skipped and only (A) runs -- exercising the CPU fallback for batch.
//
// Cases:
//   (hidden=64,   ffn_hidden=128,  n_tokens=1)
//   (hidden=2560, ffn_hidden=9216, n_tokens=1)   // Qwen3.5-4B production shape
//   (hidden=2560, ffn_hidden=9216, n_tokens=8)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct test_case {
    int64_t hidden;
    int64_t ffn_hidden;
    int64_t n_tokens;
};

static std::vector<float> make_random_f32(size_t n, uint64_t seed) {
    std::vector<float> v(n);
    std::mt19937 rng((uint32_t) seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & x : v) {
        x = dist(rng);
    }
    return v;
}

// Quantize a row-major F32 matrix [n_per_row, nrows] into TQ4_1S bytes.
// TQ4_1S is imatrix-aware; a uniform ones vector (length n_per_row) is the
// neutral choice for a synthetic test.
static void quantize_tq4(const float * src, void * dst, int64_t n_per_row, int64_t nrows) {
    std::vector<float> imatrix(n_per_row, 1.0f);
    ggml_quantize_chunk(GGML_TYPE_TQ4_1S, src, dst, 0, nrows, n_per_row, imatrix.data());
}

enum graph_kind { GK_REF, GK_FUSED };

// Build either the reference decomposition or the fused op in `ctx`.
static ggml_tensor * build_graph(ggml_context * ctx, graph_kind kind,
                                 ggml_tensor * gate, ggml_tensor * x,
                                 ggml_tensor * up, ggml_tensor * down) {
    ggml_tensor * out;
    if (kind == GK_REF) {
        ggml_tensor * t1  = ggml_mul_mat(ctx, gate, x);
        ggml_tensor * t2  = ggml_silu(ctx, t1);
        ggml_tensor * t3  = ggml_mul_mat(ctx, up, x);
        ggml_tensor * t4  = ggml_mul(ctx, t2, t3);
        out = ggml_mul_mat(ctx, down, t4);
    } else {
        out = ggml_fused_ffn(ctx, gate, x, up, down);
    }
    ggml_set_output(out);
    return out;
}

struct built_graph {
    ggml_context * ctx;
    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * down;
    ggml_tensor * x;
    ggml_tensor * out;
    ggml_cgraph * gf;
};

static built_graph make_graph(graph_kind kind, int64_t H, int64_t F, int64_t T) {
    built_graph g{};
    size_t mem = 1 << 20;
    struct ggml_init_params ip = { mem, NULL, true };
    g.ctx = ggml_init(ip);
    g.gate = ggml_new_tensor_2d(g.ctx, GGML_TYPE_TQ4_1S, H, F);
    g.up   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_TQ4_1S, H, F);
    g.down = ggml_new_tensor_2d(g.ctx, GGML_TYPE_TQ4_1S, F, H);
    g.x    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32,   H, T);
    g.out  = build_graph(g.ctx, kind, g.gate, g.x, g.up, g.down);
    g.gf   = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

static std::vector<float> compute_on(ggml_backend_t backend, graph_kind kind,
                                     int64_t H, int64_t F, int64_t T,
                                     const std::vector<uint8_t> & gate_q,
                                     const std::vector<uint8_t> & up_q,
                                     const std::vector<uint8_t> & down_q,
                                     const std::vector<uint8_t> & x_bytes) {
    // Fresh graph + context per compute: avoids stale buffer bindings when the
    // same logical op is driven across different backends in one process.
    built_graph g = make_graph(kind, H, F, T);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_gallocr_t allocr = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "  gallocr_alloc_graph FAILED\n");
        ggml_gallocr_free(allocr);
        ggml_free(g.ctx);
        return {};
    }
    ggml_backend_tensor_set(g.gate, gate_q.data(),  0, gate_q.size());
    ggml_backend_tensor_set(g.up,   up_q.data(),    0, up_q.size());
    ggml_backend_tensor_set(g.down, down_q.data(),  0, down_q.size());
    ggml_backend_tensor_set(g.x,    x_bytes.data(), 0, x_bytes.size());
    enum ggml_status st = ggml_backend_graph_compute(backend, g.gf);
    std::vector<float> res;
    if (st == GGML_STATUS_SUCCESS) {
        res.resize(ggml_nelements(g.out));
        ggml_backend_tensor_get(g.out, res.data(), 0, ggml_nbytes(g.out));
    } else {
        fprintf(stderr, "  graph_compute FAILED\n");
    }
    ggml_gallocr_free(allocr);
    ggml_free(g.ctx);
    return res;
}

static ggml_backend_t find_gpu_backend(ggml_tensor * op, std::string & name) {
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!ggml_backend_dev_supports_op(dev, op)) {
            continue;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.type == GGML_BACKEND_DEVICE_TYPE_GPU || props.type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            name = props.name;
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Peak-magnitude of the reference output, used to turn an absolute diff into a
// scale-relative one. FFN outputs span O(10)..O(1e4); an absolute tolerance is
// meaningless here, so we report |a-b|_max / |ref|_max.
static double peak_abs(const std::vector<float> & v) {
    double m = 0.0;
    for (float x : v) {
        double a = std::fabs((double) x);
        if (a > m) m = a;
    }
    return m;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;

    std::vector<test_case> cases = {
        { 64,   128,  1 },
        { 2560, 9216, 1 },
        { 2560, 9216, 8 },
    };

    const double tol = 5e-2;
    int npass_cmp = 0, ncmp = 0, nfail = 0;

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    if (!cpu_be) {
        fprintf(stderr, "FAIL: cannot init CPU backend\n");
        return 1;
    }

    for (const auto & tc : cases) {
        const int64_t H = tc.hidden, F = tc.ffn_hidden, T = tc.n_tokens;
        printf("\ncase: hidden=%lld ffn_hidden=%lld n_tokens=%lld\n",
               (long long) H, (long long) F, (long long) T);

        auto x_src    = make_random_f32(H * T, 0x1234);
        auto gate_src = make_random_f32(H * F, 0x2345);
        auto up_src   = make_random_f32(H * F, 0x3456);
        auto down_src = make_random_f32(F * H, 0x4567);

        const size_t gate_bytes = ggml_row_size(GGML_TYPE_TQ4_1S, H) * F;
        const size_t down_bytes = ggml_row_size(GGML_TYPE_TQ4_1S, F) * H;
        std::vector<uint8_t> gate_q(gate_bytes);
        std::vector<uint8_t> up_q  (gate_bytes);
        std::vector<uint8_t> down_q(down_bytes);
        quantize_tq4(gate_src.data(), gate_q.data(), H, F);
        quantize_tq4(up_src  .data(), up_q  .data(), H, F);
        quantize_tq4(down_src.data(), down_q.data(), F, H);
        std::vector<uint8_t> x_bytes(x_src.size() * sizeof(float));
        memcpy(x_bytes.data(), x_src.data(), x_bytes.size());

        // Probe GPU support with a throwaway fused graph (need the op tensor).
        std::string gname;
        ggml_backend_t gpu_be = nullptr;
        {
            built_graph probe = make_graph(GK_FUSED, H, F, T);
            gpu_be = find_gpu_backend(probe.out, gname);
            ggml_free(probe.ctx);
        }

        // 1. CPU reference decomposition.
        auto ref = compute_on(cpu_be, GK_REF, H, F, T, gate_q, up_q, down_q, x_bytes);

        // 2. CPU fused op (fallback).
        auto cpu_fused = compute_on(cpu_be, GK_FUSED, H, F, T, gate_q, up_q, down_q, x_bytes);

        if (ref.empty() || cpu_fused.empty()) {
            fprintf(stderr, "  FAIL: CPU compute returned empty\n");
            nfail++;
            continue;
        }

        // (A) CPU-fused vs CPU-ref: fallback semantics (same decomposition -> ~exact).
        double dA = max_abs_diff(cpu_fused, ref);
        bool okA = dA < tol;
        printf("  (A) CPU-fused vs CPU-ref : %s  max_abs=%.6f\n", okA ? "PASS" : "FAIL", dA);
        ncmp++; if (okA) npass_cmp++;

        // 3. Vulkan fused op (kernel), if a GPU supports this shape.
        if (gpu_be) {
            auto vk_fused = compute_on(gpu_be, GK_FUSED, H, F, T, gate_q, up_q, down_q, x_bytes);
            ggml_backend_free(gpu_be);
            if (vk_fused.empty()) {
                fprintf(stderr, "  (B) Vulkan-fused compute returned empty\n");
                nfail++;
            } else {
                // (B) Vulkan-fused vs CPU-fused: kernel correctness. FFN outputs
                // are large, so judge by peak-relative error, not absolute.
                const double dB_abs = max_abs_diff(vk_fused, cpu_fused);
                const double peak   = peak_abs(cpu_fused);
                const double dB_rel = peak > 1e-6 ? dB_abs / peak : dB_abs;
                bool okB = dB_rel < tol;
                printf("  (B) VK-fused  vs CPU-fused: %s  max_abs=%.4f peak=%.4f rel=%.4f  [%s]\n",
                       okB ? "PASS" : "FAIL", dB_abs, peak, dB_rel, gname.c_str());
                ncmp++; if (okB) npass_cmp++;
                if (!okB) {
                    printf("      vk[0..3]=%.4f %.4f %.4f %.4f | cpu[0..3]=%.4f %.4f %.4f %.4f\n",
                           vk_fused[0], vk_fused.size() > 1 ? vk_fused[1] : 0,
                           vk_fused.size() > 2 ? vk_fused[2] : 0, vk_fused.size() > 3 ? vk_fused[3] : 0,
                           cpu_fused[0], cpu_fused.size() > 1 ? cpu_fused[1] : 0,
                           cpu_fused.size() > 2 ? cpu_fused[2] : 0, cpu_fused.size() > 3 ? cpu_fused[3] : 0);
                    nfail++;
                }
            }
        } else {
            printf("  (B) VK-fused  vs CPU-fused: SKIP (no GPU supports fused_ffn for this shape)\n");
        }
    }

    ggml_backend_free(cpu_be);
    ggml_quantize_free();

    printf("\n%d/%d comparisons passed, %d failures\n", npass_cmp, ncmp, nfail);
    if (nfail > 0 || npass_cmp != ncmp) {
        fprintf(stderr, "test-fused-ffn: FAILURES\n");
        return 1;
    }
    printf("test-fused-ffn: all passed\n");
    return 0;
}
