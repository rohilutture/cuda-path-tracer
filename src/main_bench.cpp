// pt_bench: BVH build cost and closest-hit throughput vs triangle count (CPU, and GPU if built with CUDA).
//   pt_bench [--max-subdiv 7] [--rays 4000000] [--threads N] [--no-gpu] [--csv results/bench.csv]
// Rays: 'primary' (coherent, 1280x720 camera rays) and 'incoherent' (random origin, random direction).
// BVH build time is wall time of the single-threaded host SAH builder.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "pt/render_cpu.h"
#include "pt/scenes.h"
#ifdef PT_HAS_CUDA
#include "pt/gpu_renderer.h"
#endif

using namespace pt;

int main(int argc, char** argv) {
  int max_subdiv = 7, threads = (int)std::thread::hardware_concurrency();
  uint32_t nrays = 4'000'000;
  bool gpu_on = true;
  std::string csv;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
    if (arg("--max-subdiv")) max_subdiv = std::atoi(argv[++i]);
    else if (arg("--rays")) nrays = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    else if (arg("--threads")) threads = std::atoi(argv[++i]);
    else if (arg("--csv")) csv = argv[++i];
    else if (!std::strcmp(argv[i], "--no-gpu")) gpu_on = false;
  }
#ifndef PT_HAS_CUDA
  gpu_on = false;
#endif
  FILE* f = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
  const char* hdr = "tris,build_ms,sah_cost,depth,impl,ray_type,Mrays_per_s,hit_frac\n";
  std::printf("CPU threads: %d, %u rays per measurement (median of 3 CPU / 5 GPU passes)\n\n", threads, nrays);
  std::printf("| tris | build ms | SAH | depth | impl | ray type | Mrays/s | hit%% |\n|---|---|---|---|---|---|---|---|\n");
  if (f) std::fputs(hdr, f);

  auto report = [&](size_t tris, double build, double sah, int depth, const char* impl, const char* ray, const RayBenchResult& r) {
    const double hit_frac = (double)r.hits / r.rays;  // r.rays: brute-force runs use fewer rays than the BVH runs
    std::printf("| %zu | %.1f | %.1f | %d | %s | %s | %.1f | %.0f |\n", tris, build, sah, depth, impl, ray, r.mrays, 100.0 * hit_frac);
    if (f) std::fprintf(f, "%zu,%.2f,%.2f,%d,%s,%s,%.3f,%.4f\n", tris, build, sah, depth, impl, ray, r.mrays, hit_frac);
  };

  for (int sub = 2; sub <= max_subdiv; ++sub) {
    SceneAndCamera sc = noisy_scene(sub, 16.0f / 9.0f);
    const Bounds b = scene_bounds(sc.scene);
    const SceneView v = sc.scene.view();
    const size_t n = sc.scene.tris.size();
    const BuildStats& bs = sc.scene.build_stats;
    for (RayMode m : {kPrimary, kIncoherent}) {
      const char* rn = m == kPrimary ? "primary" : "incoherent";
      report(n, sc.scene.build_ms, bs.sah_cost, bs.max_depth, "cpu-bvh", rn, bench_rays_cpu(v, m, sc.camera, b, nrays, threads, false));
      if (n <= 20500) {
        // brute force is slow, so it traces only the first 200K rays; give the BVH the identical ray subset so the
        // two are directly comparable (the first primary rays are the sky-heavy top of the image: lower hit rate)
        const uint32_t sub = std::min<uint32_t>(nrays, 200000);
        report(n, sc.scene.build_ms, bs.sah_cost, bs.max_depth, "cpu-bvh-same-rays", rn, bench_rays_cpu(v, m, sc.camera, b, sub, threads, false));
        report(n, sc.scene.build_ms, bs.sah_cost, bs.max_depth, "cpu-brute", rn, bench_rays_cpu(v, m, sc.camera, b, sub, threads, true));
      }
#ifdef PT_HAS_CUDA
      if (gpu_on) {
        GpuRenderer gpu(sc.scene);
        report(n, sc.scene.build_ms, bs.sah_cost, bs.max_depth, "gpu-bvh", rn, gpu.bench_rays(m, sc.camera, b, nrays, false));
        if (n <= 20500) {
          const uint32_t sub = std::min<uint32_t>(nrays, 200000);
          report(n, sc.scene.build_ms, bs.sah_cost, bs.max_depth, "gpu-bvh-same-rays", rn, gpu.bench_rays(m, sc.camera, b, sub, false));
          report(n, sc.scene.build_ms, bs.sah_cost, bs.max_depth, "gpu-brute", rn, gpu.bench_rays(m, sc.camera, b, sub, true));
        }
      }
#endif
    }
  }
  if (f) std::fclose(f);
  return 0;
}
