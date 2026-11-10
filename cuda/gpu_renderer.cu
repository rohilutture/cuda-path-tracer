#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "pt/gpu_renderer.h"

#define CUDA_CHECK(x)                                                                         \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(1);                                                                           \
    }                                                                                         \
  } while (0)

namespace pt {

namespace {

__global__ void render_kernel(SceneView s, Camera cam, RenderSettings rs, Vec3* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= rs.width || y >= rs.height) return;
  out[(size_t)y * rs.width + x] = render_pixel(s, cam, rs, x, y);
}

// Grid-stride closest-hit kernel. Per-thread hit counts are reduced with one atomic per thread.
__global__ void ray_kernel(SceneView s, Camera cam, Bounds b, RayMode mode, int w, int h, uint32_t n, bool brute,
                           unsigned long long* hits) {
  unsigned long long local = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const Ray r = bench_ray(mode, cam, b, i, w, h, 12345);
    Hit hit;
    local += brute ? brute_intersect(s.tris, s.num_tris, r, hit) : bvh_intersect(s.nodes, s.num_nodes, s.tris, r, hit);
  }
  atomicAdd(hits, local);
}

}  // namespace

struct GpuRenderer::Impl {
  SceneView dev{};
  BVHNode* nodes = nullptr;
  Tri* tris = nullptr;
  uint32_t* mat = nullptr;
  Material* materials = nullptr;
  int sm_count = 0;
  std::string name;
};

GpuRenderer::GpuRenderer(const Scene& scene) : impl_(new Impl) {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  impl_->name = p.name;
  impl_->sm_count = p.multiProcessorCount;
  auto up = [](auto*& d, const auto& v) {
    using T = std::remove_pointer_t<std::remove_reference_t<decltype(d)>>;
    if (v.empty()) { d = nullptr; return; }
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  };
  up(impl_->nodes, scene.nodes);
  up(impl_->tris, scene.tris);
  up(impl_->mat, scene.mat);
  up(impl_->materials, scene.materials);
  impl_->dev = {impl_->nodes, impl_->tris, impl_->mat, impl_->materials, (uint32_t)scene.nodes.size(),
                (uint32_t)scene.tris.size(), scene.env};
}

GpuRenderer::~GpuRenderer() {
  if (!impl_) return;
  cudaFree(impl_->nodes); cudaFree(impl_->tris); cudaFree(impl_->mat); cudaFree(impl_->materials);
}

std::string GpuRenderer::device_name() const { return impl_->name; }

std::vector<Vec3> GpuRenderer::render(const Camera& cam, const RenderSettings& rs, float* kernel_ms) {
  Vec3* d_out;
  const size_t n = (size_t)rs.width * rs.height;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Vec3)));
  const dim3 block(16, 8), grid((rs.width + block.x - 1) / block.x, (rs.height + block.y - 1) / block.y);
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  render_kernel<<<grid, block>>>(impl_->dev, cam, rs, d_out);
  cudaEventRecord(b);
  CUDA_CHECK(cudaEventSynchronize(b));
  CUDA_CHECK(cudaGetLastError());
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  if (kernel_ms) *kernel_ms = ms;
  std::vector<Vec3> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return out;
}

RayBenchResult GpuRenderer::bench_rays(RayMode mode, const Camera& cam, const Bounds& b, uint32_t nrays, bool brute,
                                       int iters) {
  unsigned long long* d_hits;
  CUDA_CHECK(cudaMalloc(&d_hits, sizeof(unsigned long long)));
  const int threads = 128, blocks = std::min<uint32_t>((nrays + threads - 1) / threads, (uint32_t)impl_->sm_count * 32);
  cudaEvent_t a, e;
  cudaEventCreate(&a); cudaEventCreate(&e);
  std::vector<float> times;
  unsigned long long hits = 0;
  for (int it = -1; it < iters; ++it) {  // iteration -1 is a warmup
    CUDA_CHECK(cudaMemset(d_hits, 0, sizeof(unsigned long long)));
    cudaEventRecord(a);
    ray_kernel<<<blocks, threads>>>(impl_->dev, cam, b, mode, 1280, 720, nrays, brute, d_hits);
    cudaEventRecord(e);
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    cudaEventElapsedTime(&ms, a, e);
    if (it >= 0) times.push_back(ms);
  }
  CUDA_CHECK(cudaMemcpy(&hits, d_hits, sizeof(hits), cudaMemcpyDeviceToHost));
  cudaFree(d_hits);
  cudaEventDestroy(a); cudaEventDestroy(e);
  std::sort(times.begin(), times.end());
  RayBenchResult r;
  r.ms = times[times.size() / 2];
  r.mrays = nrays / (r.ms * 1e3);
  r.hits = hits;
  r.rays = nrays;
  return r;
}

}  // namespace pt
