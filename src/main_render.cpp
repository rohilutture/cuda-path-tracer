// pt_render: render a scene on the CPU and/or GPU and write a PNG.
//   pt_render --scene cornell --w 640 --h 640 --spp 256 --device cpu --out out/cornell.png
//   pt_render --scene noisy --subdiv 6 --device both          (GPU only if built with CUDA)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

#include "pt/image.h"
#include "pt/render_cpu.h"
#include "pt/scenes.h"
#ifdef PT_HAS_CUDA
#include "pt/gpu_renderer.h"
#endif

using namespace pt;

int main(int argc, char** argv) {
  std::string scene = "cornell", device = "cpu", out = "out/render.png";
  int w = 512, h = 512, spp = 64, depth = 8, subdiv = 5, threads = (int)std::thread::hardware_concurrency();
  uint32_t tris = 100000;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
    if (arg("--scene")) scene = argv[++i];
    else if (arg("--device")) device = argv[++i];
    else if (arg("--out")) out = argv[++i];
    else if (arg("--w")) w = std::atoi(argv[++i]);
    else if (arg("--h")) h = std::atoi(argv[++i]);
    else if (arg("--spp")) spp = std::atoi(argv[++i]);
    else if (arg("--depth")) depth = std::atoi(argv[++i]);
    else if (arg("--subdiv")) subdiv = std::atoi(argv[++i]);
    else if (arg("--tris")) tris = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    else if (arg("--threads")) threads = std::atoi(argv[++i]);
    else if (arg("--seed")) seed = std::strtoull(argv[++i], nullptr, 10);
  }
  const float aspect = (float)w / h;
  SceneAndCamera sc = scene == "cornell" ? cornell_box(aspect)
                    : scene == "noisy"   ? noisy_scene(subdiv, aspect)
                                         : triangle_soup(tris, seed, aspect);
  std::printf("scene %s: %zu triangles, BVH %zu nodes (depth %d, build %.1f ms, SAH cost %.1f)\n", scene.c_str(),
              sc.scene.tris.size(), sc.scene.nodes.size(), sc.scene.build_stats.max_depth, sc.scene.build_ms,
              sc.scene.build_stats.sah_cost);
  const RenderSettings rs{w, h, spp, depth, seed};
  std::vector<Vec3> cpu_img, gpu_img;
  const uint64_t paths = (uint64_t)w * h * spp;

  if (device == "cpu" || device == "both") {
    const auto t0 = std::chrono::steady_clock::now();
    cpu_img = render_cpu(sc.scene.view(), sc.camera, rs, threads);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("CPU (%d threads): %.0f ms, %.2f Mpaths/s\n", threads, ms, paths / (ms * 1e3));
  }
#ifdef PT_HAS_CUDA
  if (device == "gpu" || device == "both") {
    GpuRenderer gpu(sc.scene);
    float ms = 0;
    gpu_img = gpu.render(sc.camera, rs, &ms);
    std::printf("GPU (%s): %.1f ms kernel, %.2f Mpaths/s\n", gpu.device_name().c_str(), ms, paths / (ms * 1e3));
  }
#else
  if (device == "gpu" || device == "both") std::printf("built without CUDA: skipping the GPU render\n");
#endif
  if (!cpu_img.empty() && !gpu_img.empty()) {
    double se = 0, mean = 0;
    for (size_t i = 0; i < cpu_img.size(); ++i) {
      const Vec3 d = cpu_img[i] - gpu_img[i];
      se += dot(d, d);
      mean += dot(cpu_img[i], make3(1, 1, 1));
    }
    std::printf("CPU vs GPU: RMSE %.5f over mean pixel value %.4f (same algorithm and random streams)\n",
                std::sqrt(se / (3.0 * cpu_img.size())), mean / (3.0 * cpu_img.size()));
  }
  const auto& img = !gpu_img.empty() ? gpu_img : cpu_img;
  if (!img.empty()) {
    const auto parent = std::filesystem::path(out).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::printf(write_png(out, w, h, tonemap(img)) ? "wrote %s\n" : "FAILED to write %s\n", out.c_str());
  }
  return 0;
}
