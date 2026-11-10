#pragma once
// The path tracer. Everything here is PT_HD: the CPU reference renderer and the CUDA kernel both call
// render_pixel(), so they run the same algorithm with the same per-pixel random streams.
//
// Unidirectional path tracing with cosine-weighted hemisphere sampling (so a diffuse bounce multiplies
// throughput by the albedo exactly), perfect mirrors, emission on any surface, a constant environment, and
// Russian roulette after depth 3. Unbiased: the white-furnace tests check this against closed-form answers.
#include "camera.h"
#include "scene.h"

namespace pt {

struct RenderSettings {
  int width, height, spp, max_depth;
  uint64_t seed;
};

// Orthonormal basis around n (Duff et al. 2017, branchless).
PT_HD inline void basis(Vec3 n, Vec3& t, Vec3& b) {
  const float s = copysignf(1.0f, n.z);
  const float a = -1.0f / (s + n.z);
  const float c = n.x * n.y * a;
  t = {1.0f + s * n.x * n.x * a, s * c, -s * n.x};
  b = {c, s + n.y * n.y * a, -n.y};
}

// Cosine-weighted direction in the hemisphere around n: pdf = cos(theta) / pi.
PT_HD inline Vec3 sample_cosine(Vec3 n, float u1, float u2) {
  const float r = sqrtf(u1), phi = 2.0f * kPi * u2;
  const float x = r * cosf(phi), y = r * sinf(phi), z = sqrtf(fmaxf(0.0f, 1.0f - u1));
  Vec3 t, b;
  basis(n, t, b);
  return normalize(t * x + b * y + n * z);
}

PT_HD inline Vec3 trace(const SceneView& s, Ray r, Rng& rng, int max_depth) {
  Vec3 L = {0, 0, 0}, thr = {1, 1, 1};
  for (int depth = 0; depth < max_depth; ++depth) {
    Hit h;
    if (!bvh_intersect(s.nodes, s.num_nodes, s.tris, r, h)) {
      L += thr * s.env;
      break;
    }
    const Material& m = s.materials[s.mat[h.tri]];
    L += thr * m.emission;
    Vec3 n = tri_normal(s.tris[h.tri]);
    if (dot(n, r.d) > 0.0f) n = -n;  // face the incoming ray (surfaces are double-sided)
    const Vec3 p = r.o + r.d * h.t;
    Vec3 dir;
    if (m.type == kMirror) {
      dir = r.d - n * (2.0f * dot(r.d, n));
      thr = thr * m.albedo;
    } else {
      // Draw sequentially: C++ leaves argument evaluation order unspecified, and gcc and nvcc differ, which would
      // silently desynchronize the CPU and GPU random streams from the first diffuse bounce on.
      const float u1 = rng.next_f();
      const float u2 = rng.next_f();
      dir = sample_cosine(n, u1, u2);
      thr = thr * m.albedo;  // brdf * cos / pdf = albedo
    }
    if (depth >= 3) {  // Russian roulette: survive with probability q, divide by q to stay unbiased
      const float q = fminf(0.95f, max_component(thr));
      if (q <= 0.0f || rng.next_f() >= q) break;
      thr = thr * (1.0f / q);
    }
    r = {p, dir, 1e-4f, kInf};
  }
  return L;
}

// One pixel, `spp` samples with jittered positions. The random stream is a pure function of (seed, pixel).
PT_HD inline Vec3 render_pixel(const SceneView& s, const Camera& cam, const RenderSettings& rs, int x, int y) {
  Rng rng(rs.seed, (uint64_t)y * rs.width + x);
  Vec3 acc = {0, 0, 0};
  for (int i = 0; i < rs.spp; ++i) {
    const float u = (x + rng.next_f()) / rs.width;
    const float v = (y + rng.next_f()) / rs.height;
    acc += trace(s, cam.ray(u, 1.0f - v), rng, rs.max_depth);
  }
  return acc * (1.0f / rs.spp);
}

}  // namespace pt
