// ponytail: build-time consistency guard for TQ2_1S centroid + sign tables.
//
// Ceiling: this is a path-relative source-tree test. It fopens the 4 source
// files (paths baked at build time via compile defs), regex-extracts the
// centroid arrays by stable anchor tokens, and parses the floats/signed
// values numerically. It BREAKS if a .comp / .c file moves, or if an array
// literal is refactored so the anchor token no longer precedes the values.
// That brittleness is the point: any single-file centroid/sign edit that
// forgets a sibling file is caught here instead of producing silent garbage
// output at runtime (the bug class that gibberished TQ2_1S in the first place).
//
// If shaders move to a generated/embedded form, replace the fopen path with a
// shared #include header (tq_tables.glslh) included by all .comp files and emit
// the same values into a C header; this test then asserts C == header.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef TQ_QUANT_C
#error "TQ_QUANT_C source path compile def required"
#endif

static bool read_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n);
    size_t got = fread(&out[0], 1, n, f);
    fclose(f);
    out.resize(got);
    return true;
}

// From `src` starting at `pos`, scan forward to the first '{' or '(',
// then read up to `maxn` numeric tokens (optionally signed floats) until
// the matching '}' or ')'. Returns parsed floats.
static std::vector<double> parse_array(const std::string & src, size_t anchor, int maxn) {
    std::vector<double> out;
    size_t i = anchor;
    while (i < src.size() && src[i] != '{' && src[i] != '(') i++;
    if (i >= src.size()) return out;
    char term = (src[i] == '{') ? '}' : ')';
    i++;
    while (i < src.size() && src[i] != term && (int)out.size() < maxn) {
        char * endp = nullptr;
        double v = strtod(src.c_str() + i, &endp);
        if (endp == src.c_str() + i) { i++; continue; }  // not a number (comment/comma)
        out.push_back(v);
        i = endp - src.c_str();
    }
    return out;
}

static size_t find_anchor(const std::string & src, const char * anchor) {
    size_t p = src.find(anchor);
    return p;
}

static int fail(const char * msg) {
    fprintf(stderr, "FAIL test-tq-tables: %s\n", msg);
    return 1;
}

static bool approx(const std::vector<double> & got, const std::vector<double> & want, double tol) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < got.size(); i++) {
        if (fabs(got[i] - want[i]) > tol) return false;
    }
    return true;
}

int main() {
    // ---- Centroids: Lloyd-Max {+-0.4528, +-1.5104} everywhere (unit-sigma), ----
    // scaled variants in the two shaders that pre-multiply by a constant.
    const std::vector<double> lm_centroids = {-1.5104, -0.4528, 0.4528, 1.5104};
    // matvec pre-multiplies by 1/sqrt(32) = 0.17677669529663688
    const std::vector<double> lm_scaled_sqrt32 = {
        -1.5104 * 0.17677669529663688, -0.4528 * 0.17677669529663688,
         0.4528 * 0.17677669529663688,  1.5104 * 0.17677669529663688,
    };
    // copy_to_quant uses 1/sqrt(128) scale: {+-0.039994, +-0.133462}
    const std::vector<double> lm_scaled_sqrt128 = {-0.133462, -0.039994, 0.039994, 0.133462};

    const double tol = 1e-4;

    // --- 1. C source: TQ2_1S_CENTROIDS ---
    {
        std::string s;
        if (!read_file(TQ_QUANT_C, s)) return fail("cannot open ggml-turbo-quant.c");
        size_t a = find_anchor(s, "TQ2_1S_CENTROIDS");
        if (a == std::string::npos) return fail("TQ2_1S_CENTROIDS anchor missing");
        auto got = parse_array(s, a, 4);
        if (!approx(got, lm_centroids, tol)) return fail("C TQ2_1S_CENTROIDS != Lloyd-Max");
    }

#ifdef TQ_DEQUANT_COMP
    // --- 2. dequant shader: centroids[4] ---
    {
        std::string s;
        if (!read_file(TQ_DEQUANT_COMP, s)) return fail("cannot open dequant_tq2_1s.comp");
        size_t a = find_anchor(s, "centroids[4]");
        if (a == std::string::npos) return fail("dequant centroids[4] anchor missing");
        auto got = parse_array(s, a, 4);
        if (!approx(got, lm_centroids, tol)) return fail("dequant centroids != Lloyd-Max");
    }
#endif

#ifdef TQ_MATVEC_COMP
    // --- 3. matvec shader: TQ2_CENTROIDS_SCALED[4] (pre-mult 1/sqrt(32)) ---
    {
        std::string s;
        if (!read_file(TQ_MATVEC_COMP, s)) return fail("cannot open mul_mat_vec_tq2_1s.comp");
        size_t a = find_anchor(s, "TQ2_CENTROIDS_SCALED");
        if (a == std::string::npos) return fail("matvec TQ2_CENTROIDS_SCALED anchor missing");
        auto got = parse_array(s, a, 4);
        if (!approx(got, lm_scaled_sqrt32, tol)) return fail("matvec TQ2_CENTROIDS_SCALED != Lloyd-Max/sqrt(32)");
    }
#endif

#ifdef TQ_MULMM_COMP
    // --- 4. fused mul_mm shader: TQ2_CENTROIDS[4] (unscaled; B-side WHT carries 1/sqrt(32)) ---
    {
        std::string s;
        if (!read_file(TQ_MULMM_COMP, s)) return fail("cannot open mul_mm_tq2_1s.comp");
        size_t a = find_anchor(s, "TQ2_CENTROIDS[4]");
        if (a == std::string::npos) return fail("mul_mm TQ2_CENTROIDS[4] anchor missing");
        auto got = parse_array(s, a, 4);
        if (!approx(got, lm_centroids, tol)) return fail("mul_mm TQ2_CENTROIDS != Lloyd-Max");
    }
#endif

#ifdef TQ_COPY_COMP
    // --- 5. copy_to_quant shader: TC2 (Lloyd-Max / sqrt(128)) ---
    {
        std::string s;
        if (!read_file(TQ_COPY_COMP, s)) return fail("cannot open copy_to_quant.comp");
        size_t a = find_anchor(s, "TC2[4]");
        if (a == std::string::npos) return fail("copy_to_quant TC2[4] anchor missing");
        auto got = parse_array(s, a, 4);
        if (!approx(got, lm_scaled_sqrt128, tol)) return fail("copy_to_quant TC2 != Lloyd-Max/sqrt(128)");
    }
#endif

    // ---- Sign tables: 32-element RHT sign vector, must be byte-identical ----
    // numerically across C (TQ3_0_SIGNS), dequant (signs), matvec (TQ2_SIGNS).
    // Parse numerically (C uses "+1.0f", GLSL uses "+1.0" — byte compare would lie).
#ifdef TQ_DEQUANT_COMP
    std::vector<double> sig_c, sig_deq, sig_mv;
    {
        std::string s;
        if (!read_file(TQ_QUANT_C, s)) return fail("cannot open ggml-turbo-quant.c (signs)");
        size_t a = find_anchor(s, "TQ3_0_SIGNS");
        if (a == std::string::npos) return fail("C TQ3_0_SIGNS anchor missing");
        sig_c = parse_array(s, a, 32);
        if ((int)sig_c.size() != 32) return fail("C TQ3_0_SIGNS did not yield 32 values");
    }
    {
        std::string s;
        if (!read_file(TQ_DEQUANT_COMP, s)) return fail("cannot open dequant (signs)");
        size_t a = find_anchor(s, "signs[32]");
        if (a == std::string::npos) return fail("dequant signs[32] anchor missing");
        sig_deq = parse_array(s, a, 32);
        if ((int)sig_deq.size() != 32) return fail("dequant signs[32] did not yield 32 values");
    }
    {
        std::string s;
        if (!read_file(TQ_MATVEC_COMP, s)) return fail("cannot open matvec (signs)");
        size_t a = find_anchor(s, "TQ2_SIGNS[32]");
        if (a == std::string::npos) return fail("matvec TQ2_SIGNS[32] anchor missing");
        sig_mv = parse_array(s, a, 32);
        if ((int)sig_mv.size() != 32) return fail("matvec TQ2_SIGNS[32] did not yield 32 values");
    }
#ifdef TQ_MULMM_COMP
    std::vector<double> sig_mm;
    {
        std::string s;
        if (!read_file(TQ_MULMM_COMP, s)) return fail("cannot open mul_mm (signs)");
        size_t a = find_anchor(s, "TQ_SIGNS[32]");
        if (a == std::string::npos) return fail("mul_mm TQ_SIGNS[32] anchor missing");
        sig_mm = parse_array(s, a, 32);
        if ((int)sig_mm.size() != 32) return fail("mul_mm TQ_SIGNS[32] did not yield 32 values");
    }
#endif
    for (int i = 0; i < 32; i++) {
        if (sig_c[i] != sig_deq[i]) return fail("C vs dequant sign mismatch");
        if (sig_c[i] != sig_mv[i])  return fail("C vs matvec sign mismatch");
#ifdef TQ_MULMM_COMP
        if (sig_c[i] != sig_mm[i])  return fail("C vs mul_mm sign mismatch");
#endif
        if (sig_c[i] != 1.0 && sig_c[i] != -1.0) return fail("sign value not +-1");
    }
#endif

    printf("ok test-tq-tables: TQ2_1S centroids (Lloyd-Max) + signs consistent across C/dequant/matvec/mul_mm/copy\n");
    return 0;
}
