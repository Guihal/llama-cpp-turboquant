# llama-cpp-turboquant

Fork of llama.cpp with TurboQuant KV cache quantization.

## Build
```
cmake -B build-vulkan -DGGML_VULKAN=ON
cmake --build build-vulkan -j 6
```

## Hardware target
Radeon 760M (RDNA3, 8 CUs, 8.5 GB shared VRAM), 14 GB DDR5, Ryzen 5 7640HS.

## TQ weight types (NOT on Vulkan yet — fall back to CPU)
- TQ3_1S, TQ4_1S — WHT-rotated Lloyd-Max polar quant
- Tensors on CPU at inference, Vulkan needs additional pipelines

## KV cache TQ (works on GPU)
- TURBO2_0, TURBO3_0, TURBO4_0 — in dequant_funcs_cm2.glsl

## ZAYA1-8B model
- Source: src/models/zaya.cpp
- Architecture: CCA (Compressed Convolutional Attention) + MoE++
- 40 transformer layers, 760M active / 8.4B total params
- ggml_conv_1d_grouped sub-graph in ggml/src/ggml.c

## Open tasks
- Port TQ4_1S Vulkan kernel (~10 lines C++ glue, shaders exist)
- Port TQ3_1S Vulkan kernel (from scratch)
