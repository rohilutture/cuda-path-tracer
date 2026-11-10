#pragma once
// Ray generators for pure intersection-throughput benchmarks, shared by CPU and GPU so both trace the same rays.
//   kPrimary   : one ray per pixel through the camera (coherent: neighbouring rays take similar BVH paths)
//   kIncoherent: random origin inside the scene bounds, uniform random direction. These rays are directionally
//                incoherent, but they start inside the scene and are short, so they are NOT necessarily slower than
//                primary rays (measured: they are faster on the CPU). Compare Mrays/s only within one ray type.
#include "camera.h"

namespace pt {

enum RayMode : int { kPrimary = 0, kIncoherent = 1 };

PT_HD inline Ray bench_ray(RayMode mode, const Camera& cam, const Bounds& b, uint32_t idx, int w, int h, uint64_t seed) {
  if (mode == kPrimary) {
    const uint32_t p = idx % (uint32_t)(w * h);
    const int x = (int)(p % (uint32_t)w), y = (int)(p / (uint32_t)w);
    return cam.ray((x + 0.5f) / w, 1.0f - (y + 0.5f) / h);
  }
  Rng rng(seed, idx);
  const Vec3 o = {b.lo.x + rng.next_f() * (b.hi.x - b.lo.x), b.lo.y + rng.next_f() * (b.hi.y - b.lo.y),
                  b.lo.z + rng.next_f() * (b.hi.z - b.lo.z)};
  const float z = 1.0f - 2.0f * rng.next_f(), r = sqrtf(fmaxf(0.0f, 1.0f - z * z)), phi = 2.0f * kPi * rng.next_f();
  return {o, {r * cosf(phi), r * sinf(phi), z}, 1e-4f, kInf};
}

}  // namespace pt
