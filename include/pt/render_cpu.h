#pragma once
// Multithreaded CPU reference renderer and CPU ray benchmark. Same render_pixel() as the GPU kernel.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include "bench_rays.h"
#include "pathtrace.h"

namespace pt {

inline std::vector<Vec3> render_cpu(const SceneView& s, const Camera& cam, const RenderSettings& rs, int threads) {
  std::vector<Vec3> img((size_t)rs.width * rs.height);
  std::atomic<int> next{0};
  auto worker = [&] {
    for (int y; (y = next.fetch_add(1)) < rs.height;)
      for (int x = 0; x < rs.width; ++x) img[(size_t)y * rs.width + x] = render_pixel(s, cam, rs, x, y);
  };
  std::vector<std::thread> pool;
  for (int i = 1; i < std::max(1, threads); ++i) pool.emplace_back(worker);
  worker();
  for (auto& t : pool) t.join();
  return img;
}

struct RayBenchResult {
  double ms = 0, mrays = 0;
  uint64_t hits = 0;
  uint32_t rays = 0;  // number of rays actually traced (callers may use fewer for slow variants like brute force)
};

// Median of `iters` timed passes over `nrays` closest-hit queries. brute = test every triangle (small scenes only).
inline RayBenchResult bench_rays_cpu(const SceneView& s, RayMode mode, const Camera& cam, const Bounds& b, uint32_t nrays,
                                     int threads, bool brute, int iters = 3) {
  const int w = 1280, h = 720;
  std::vector<double> times;
  uint64_t hits_total = 0;
  for (int it = 0; it < iters; ++it) {
    std::atomic<uint32_t> next{0};
    std::atomic<uint64_t> hits{0};
    auto worker = [&] {
      uint64_t local = 0;
      constexpr uint32_t kChunk = 4096;
      for (uint32_t base; (base = next.fetch_add(kChunk)) < nrays;) {
        for (uint32_t i = base; i < std::min(nrays, base + kChunk); ++i) {
          Ray r = bench_ray(mode, cam, b, i, w, h, 12345);
          Hit hit;
          local += brute ? brute_intersect(s.tris, s.num_tris, r, hit) : bvh_intersect(s.nodes, s.num_nodes, s.tris, r, hit);
        }
      }
      hits += local;
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int i = 1; i < std::max(1, threads); ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    hits_total = hits;
  }
  std::sort(times.begin(), times.end());
  RayBenchResult r;
  r.ms = times[times.size() / 2];
  r.mrays = nrays / (r.ms * 1e3);
  r.hits = hits_total;
  r.rays = nrays;
  return r;
}

}  // namespace pt
