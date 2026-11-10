# CUDA path tracer with a SAH BVH

Status: **the algorithm is verified on CPU (22 tests, sanitizer-clean) and the CUDA renderer now runs on an RTX 5090 and
reproduces the CPU image (RMSE about 1e-4). First GPU benchmark below; single runs, not yet profiled or optimized.**

## GPU results (RTX 5090 from WSL2, nvcc sm_120; CPU = 24-thread Core Ultra 9 285K; `results/bench.txt`, `results/render_gpu.txt`)

**Correctness: the first GPU run exposed a real bug, now fixed.** CPU and GPU images matched exactly at one bounce but
differed by pure Monte Carlo noise from the first diffuse bounce (RMSE 0.15 at 128 spp, falling as 1/sqrt(spp)).
Cause: `sample_cosine(n, rng.next_f(), rng.next_f())` relies on function-argument evaluation order, which C++ leaves
unspecified and gcc and nvcc resolve differently, so the two random numbers were swapped between host and device.
After drawing them in separate statements the images agree: RMSE 0.00000 at 1 to 16 spp and at most 0.0005 at 64 to 512 spp
(tiny residual from rare float-level path divergence), on a mean pixel value of 0.43. `scripts/check_gpu_matches_cpu.sh`
is the regression check. The CPU tests could not see this bug.

- **Path tracing, Cornell box 640x640, 512 spp, 8 bounces:** GPU 319 ms (657 Mpaths/s) vs 14.9 s on 24 CPU threads (14 Mpaths/s),
  about 47x. Same algorithm and random streams, unoptimized thread-per-pixel kernel; the CPU number is a straightforward
  multithreaded implementation, not Embree, so this is not a claim about production CPU renderers.
- **BVH closest-hit throughput (4M rays, median of 5):** 5,122 triangles: 9,090 Mrays/s primary (CPU 60); 327,682 triangles:
  4,970 Mrays/s primary, 7,240 incoherent (CPU 48 and 159). Throughput falls about 45% from 5K to 327K triangles.
  These ray sets are generated inside the scene bounds (see the note above on why incoherent is not harder here).
- **BVH vs testing every triangle, GPU, identical rays** (lower bound, short runs): 5,122 tris 31x, 20,482 tris 91x (primary);
  322 tris only 1.8x. The same-ray runs use a small ray subset, so timing overhead depresses the BVH side.
- Not profiled: the traversal stack is local memory (272 B per thread); no Nsight data, so occupancy and divergence claims
  would be guesses. No comparison against OptiX.

Older text below that says "never run on the GPU" refers to the state before these results.

A path tracer whose core (math, ray-triangle test, BVH build and traversal, the path-tracing loop) lives in
headers marked `__host__ __device__`. The CPU reference renderer and the CUDA kernel call the *same*
`render_pixel()` with the same per-pixel PCG random streams, so CPU results are real evidence about what the
GPU will compute, and a CPU-vs-GPU image comparison is a built-in correctness check (`pt_render --device both`).

| Piece | File | Notes |
|---|---|---|
| Binned-SAH BVH builder | `include/pt/bvh.h` | 16 bins, leaf size 4, depth cap 48, host only; flattened 32-byte nodes |
| Traversal | `include/pt/bvh.h` | stack-based closest hit, ordered (near child first via stored split axis), finite reciprocals (no inf/NaN), runs on host and device |
| Path tracer | `include/pt/pathtrace.h` | unidirectional, cosine-weighted diffuse, perfect mirror, emission, constant environment, Russian roulette |
| Scenes | `include/pt/scenes.h` | Cornell box, bumpy icosphere (20*4^n triangles, for scaling), random triangle soup |
| CPU renderer + ray benchmark | `include/pt/render_cpu.h` | multithreaded, deterministic regardless of thread count |
| CUDA renderer | `cuda/gpu_renderer.cu` | one thread per pixel; also a closest-hit throughput kernel, BVH vs brute force |
| Output | `include/pt/image.h` | dependency-free PNG writer |

## Verified (CPU, 22 tests)

- **BVH == brute force**: identical hit/miss and distance on thousands of random and axis-aligned rays, for 1 to
  2000 triangles and leaf sizes 1/4/8, on an 82K-triangle bumpy sphere, and on degenerate geometry (200 identical
  triangles, concentric triangles with coincident centroids, a flat scene, zero-area triangles, an empty scene).
  Structural invariants hold (each triangle in exactly one leaf, parents bound children, depth fits the stack).
- **Energy conservation against closed forms ("white furnace")**: inside a sphere whose wall emits Le and reflects
  albedo rho, an unbiased tracer must converge to Le/(1-rho). Four (rho, Le) pairs agree within 1.2 percent; a
  depth cap of 2 gives exactly Le(1+rho); a convex albedo-1 sphere in uniform light reflects exactly 1; a mirror
  sphere reflects the environment colour exactly.
- Sampling and numerics: cosine-hemisphere moments (E[cos] = 2/3, E[cos^2] = 1/2), orthonormal basis including
  the poles, RNG mean/variance and stream independence, no cracks along a shared triangle edge, camera geometry.
- Rendering: deterministic and identical for 1 vs 4 threads; the Cornell box has a red left wall and green right
  wall; error falls as 1/sqrt(spp).

Bugs the tests found and fixed: (1) a ray starting exactly on a box face with a zero direction component produced
`0 * inf = NaN`, and min/max dropped the NaN while the `inf` poisoned the interval, so the ray missed a box it
touched (fixed with finite clamped reciprocals); (2) a first attempt at the sanitizer run silently did not run
because `stdbuf` (LD_PRELOAD) breaks the ASan runtime.

Also clean under AddressSanitizer + UBSan and ThreadSanitizer (all 22 tests).

## Measured results (CPU only)

Setup: Intel Core Ultra 9 285K (24 cores) under WSL2, gcc 13.3 `-O3 -march=native`, 24 threads. Raw output in
`results/` (`bench.txt`, `bench.csv`, `render.txt`). The bumpy-sphere scene (`noisy_scene`) is a sphere with
two octaves of value-noise displacement plus a floor, 20*4^n triangles; 4M rays per measurement, median of 3.

**BVH build** (single-threaded host SAH builder) and **BVH closest-hit throughput**, 24 threads:

| triangles | build | SAH cost | depth | primary rays | incoherent rays |
|---|---|---|---|---|---|
| 322 | 0.1 ms | 3.9 | 9 | ~90 Mrays/s | ~160-200 Mrays/s |
| 5,122 | 2.2 ms | 4.7 | 13 | ~67 | ~170 |
| 20,482 | 10 ms | 5.1 | 16 | ~55 | ~145-175 |
| 81,922 | 43 ms | 5.5 | 18 | ~55 | ~175 |
| 327,682 | 190 ms | 5.8 | 21 | ~48-55 | ~155-170 |

Throughput falls only slowly as the scene grows 1000x (depth 9 to 21), which is the point of a BVH. Do not
compare primary against incoherent: incoherent rays start at random points inside the scene bounds and are short,
so they are not harder here (they are faster); that is a property of this ray generator, not of coherence.

**BVH vs testing every triangle** (identical ray subsets; the BVH rows are a *lower* bound because a 200K-ray
run lasts only a few ms, so thread start-up is a visible share of its time):

| triangles | ray type | brute force | BVH, same rays | speedup (at least) |
|---|---|---|---|---|
| 322 | primary | 16.2 Mrays/s | 43.4 | 3x |
| 1,282 | primary | 5.5 | 36.3 | 7x |
| 5,122 | primary | 1.1 | 47.8 | 44x |
| 20,482 | primary | 0.10 | 24.1 | 234x |
| 20,482 | incoherent | 0.11 | 72.5 | 677x |

**Path tracing**: the 640x640, 512 spp, 8-bounce Cornell box (5,144 triangles, BVH depth 14) renders in about
16 s on 24 CPU threads, 12.9 Mpaths/s (`results/cornell_cpu.png`). The black patch inside the mirror sphere is the
reflection of the box's open front, whose environment is black in this scene.

### What is NOT measured

- **(Superseded: GPU now measured above; still no profiler data.)** `pt_render --device both` (prints CPU-vs-GPU RMSE) and `pt_bench` without `--no-gpu`
  exist and compile, but have never run. The CUDA kernel is a straightforward thread-per-pixel design; expect it
  to need real work (the traversal stack is 272 bytes of local memory per thread, so occupancy and divergence are
  the first things to profile), and a GPU-vs-CPU speedup is unknown.
- No comparison against OptiX/Embree; no next-event estimation, so noise in the Cornell render is high at low spp.
- Triangle meshes only; constant-colour diffuse and mirror materials, no textures, glass, or importance-sampled lights.

## Next steps

1. (Done) Run on the GPU: CPU-vs-GPU RMSE, then `pt_bench`.
2. Profile (Nsight Compute: occupancy, divergence, stack spills) and try the obvious fixes: wavefront
   (path-regeneration) kernels, a shorter stack, treelet/node layout changes, `__ldg`.
3. Add next-event estimation for the area light and multiple importance sampling; add an OBJ loader.

## Run

```bash
bash scripts/build.sh                                   # WSL; builds CPU targets and the CUDA renderer (compile only)
~/pt-build/release/pt_tests                             # 22 tests
~/pt-build/release/pt_render --scene cornell --spp 512 --device cpu --out out/cornell.png
~/pt-build/release/pt_render --scene noisy --subdiv 6 --device both     # CPU vs GPU: prints RMSE (needs a free GPU)
bash scripts/run_bench.sh [--with-gpu]
bash scripts/sanitize.sh
```

## Phrasing rules (for the resume)

- Safe now: "CUDA/C++ path tracer with a binned-SAH BVH; the core is shared host/device code so the CPU build
  verifies the algorithm; BVH validated against brute force (including degenerate geometry) and the path tracer
  against closed-form white-furnace energy tests; clean under ASan/UBSan/TSan". State that GPU performance has
  not been measured until it has.
- CPU numbers need "24-thread CPU, WSL2" with them. Never present the CPU benchmark as GPU performance, and do
  not claim a speedup over OptiX/Embree or any production renderer.
