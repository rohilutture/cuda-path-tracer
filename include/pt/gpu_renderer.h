#pragma once
// Host-side API for the CUDA renderer. No CUDA types here, so any C++ file can include it; the
// implementation (cuda/gpu_renderer.cu) is only built when a CUDA toolchain is present (PT_HAS_CUDA).
#include <memory>
#include <string>
#include <vector>

#include "bench_rays.h"
#include "render_cpu.h"

namespace pt {

class GpuRenderer {
 public:
  explicit GpuRenderer(const Scene& scene);  // uploads BVH, triangles, materials
  ~GpuRenderer();
  GpuRenderer(const GpuRenderer&) = delete;
  GpuRenderer& operator=(const GpuRenderer&) = delete;

  // Path-traced image (linear radiance). kernel_ms = GPU kernel time measured with CUDA events.
  std::vector<Vec3> render(const Camera& cam, const RenderSettings& rs, float* kernel_ms = nullptr);

  // Pure closest-hit throughput. brute = test every triangle instead of using the BVH.
  RayBenchResult bench_rays(RayMode mode, const Camera& cam, const Bounds& b, uint32_t nrays, bool brute, int iters = 5);

  std::string device_name() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pt
