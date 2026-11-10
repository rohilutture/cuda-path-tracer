#pragma once
// Procedural scenes (host only): Cornell box, icospheres (smooth and noisy, for BVH scaling), triangle soup.
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <random>
#include <utility>

#include "camera.h"
#include "scene.h"

namespace pt {

// Subdivided icosahedron projected to a sphere: 20 * 4^subdiv triangles. Returns triangles as vertex triples.
inline std::vector<std::array<Vec3, 3>> icosphere_tris(int subdiv) {
  const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
  std::vector<Vec3> v = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                         {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (auto& p : v) p = normalize(p);
  std::vector<std::array<int, 3>> f = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                                       {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
                                       {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
  for (int s = 0; s < subdiv; ++s) {
    std::map<std::pair<int, int>, int> cache;
    auto mid = [&](int a, int b) {
      auto key = std::make_pair(std::min(a, b), std::max(a, b));
      auto it = cache.find(key);
      if (it != cache.end()) return it->second;
      v.push_back(normalize((v[a] + v[b]) * 0.5f));
      return cache[key] = (int)v.size() - 1;
    };
    std::vector<std::array<int, 3>> nf;
    nf.reserve(f.size() * 4);
    for (auto& tr : f) {
      int a = mid(tr[0], tr[1]), b = mid(tr[1], tr[2]), c = mid(tr[2], tr[0]);
      nf.push_back({tr[0], a, c}); nf.push_back({tr[1], b, a}); nf.push_back({tr[2], c, b}); nf.push_back({a, b, c});
    }
    f = std::move(nf);
  }
  std::vector<std::array<Vec3, 3>> out;
  out.reserve(f.size());
  for (auto& tr : f) out.push_back({v[tr[0]], v[tr[1]], v[tr[2]]});
  return out;
}

inline void add_icosphere(Scene& s, Vec3 c, float r, int subdiv, uint32_t m) {
  for (auto& t : icosphere_tris(subdiv)) s.add_tri(c + t[0] * r, c + t[1] * r, c + t[2] * r, m);
}

// Cheap deterministic 3-D value noise for irregular test geometry.
inline float hash3(int x, int y, int z) {
  uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2147483647u;
  h = (h ^ (h >> 13)) * 1274126177u;
  return (float)((h ^ (h >> 16)) & 0xffffff) / 16777216.0f;
}
inline float value_noise(Vec3 p) {
  int xi = (int)std::floor(p.x), yi = (int)std::floor(p.y), zi = (int)std::floor(p.z);
  float fx = p.x - xi, fy = p.y - yi, fz = p.z - zi;
  auto sm = [](float t) { return t * t * (3 - 2 * t); };
  fx = sm(fx); fy = sm(fy); fz = sm(fz);
  float r = 0;
  for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx)
    r += hash3(xi + dx, yi + dy, zi + dz) * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
  return r;
}

// An irregular, bumpy sphere with ~20 * 4^subdiv triangles: realistic non-uniform triangle sizes for BVH scaling.
inline void add_noisy_sphere(Scene& s, Vec3 c, float r, int subdiv, uint32_t m, float bump = 0.25f) {
  auto disp = [&](Vec3 p) { return p * (r * (1.0f + bump * (value_noise(p * 3.0f) - 0.5f) + 0.5f * bump * (value_noise(p * 9.0f) - 0.5f))); };
  for (auto& t : icosphere_tris(subdiv)) s.add_tri(c + disp(t[0]), c + disp(t[1]), c + disp(t[2]), m);
}

inline void add_box(Scene& s, Vec3 c, Vec3 half, float yaw, uint32_t m) {
  const float cs = std::cos(yaw), sn = std::sin(yaw);
  auto P = [&](float x, float y, float z) {
    x *= half.x; y *= half.y; z *= half.z;
    return c + Vec3{cs * x + sn * z, y, -sn * x + cs * z};
  };
  const Vec3 v[8] = {P(-1, -1, -1), P(1, -1, -1), P(1, 1, -1), P(-1, 1, -1), P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1)};
  const int q[6][4] = {{0, 1, 2, 3}, {5, 4, 7, 6}, {4, 0, 3, 7}, {1, 5, 6, 2}, {3, 2, 6, 7}, {4, 5, 1, 0}};
  for (auto& f : q) s.add_quad(v[f[0]], v[f[1]], v[f[2]], v[f[3]], m);
}

struct SceneAndCamera {
  Scene scene;
  Camera camera;
};

inline SceneAndCamera cornell_box(float aspect, int sphere_subdiv = 4) {
  SceneAndCamera sc;
  Scene& s = sc.scene;
  const uint32_t white = s.add_material({0.73f, 0.73f, 0.73f}), red = s.add_material({0.65f, 0.05f, 0.05f}),
                 green = s.add_material({0.12f, 0.45f, 0.15f}),
                 light = s.add_material({0, 0, 0}, {17.0f, 14.0f, 10.0f}),
                 mirror = s.add_material({0.95f, 0.95f, 0.95f}, {0, 0, 0}, kMirror);
  s.add_quad({-1, -1, -1}, {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, white);   // floor
  s.add_quad({-1, 1, 1}, {1, 1, 1}, {1, 1, -1}, {-1, 1, -1}, white);       // ceiling
  s.add_quad({-1, -1, -1}, {-1, 1, -1}, {1, 1, -1}, {1, -1, -1}, white);   // back
  s.add_quad({-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, -1}, red);     // left
  s.add_quad({1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}, green);       // right
  s.add_quad({-0.3f, 0.995f, -0.3f}, {0.3f, 0.995f, -0.3f}, {0.3f, 0.995f, 0.3f}, {-0.3f, 0.995f, 0.3f}, light);
  add_box(s, {-0.45f, -0.4f, -0.3f}, {0.3f, 0.6f, 0.3f}, 0.4f, white);
  add_icosphere(s, {0.5f, -0.6f, 0.2f}, 0.4f, sphere_subdiv, mirror);
  s.env = {0, 0, 0};
  s.finalize();
  sc.camera = Camera::look_at({0, 0, 3.4f}, {0, 0, 0}, {0, 1, 0}, 40.0f, aspect);
  return sc;
}

// Bumpy sphere on a floor under a bright sky: scales to millions of triangles for benchmarking.
inline SceneAndCamera noisy_scene(int subdiv, float aspect) {
  SceneAndCamera sc;
  Scene& s = sc.scene;
  const uint32_t grey = s.add_material({0.7f, 0.7f, 0.7f}), blue = s.add_material({0.2f, 0.35f, 0.7f});
  add_noisy_sphere(s, {0, 0, 0}, 1.0f, subdiv, blue);
  s.add_quad({-6, -1.4f, -6}, {6, -1.4f, -6}, {6, -1.4f, 6}, {-6, -1.4f, 6}, grey);
  s.env = {0.8f, 0.9f, 1.0f};
  s.finalize();
  sc.camera = Camera::look_at({0, 0.6f, 3.6f}, {0, -0.1f, 0}, {0, 1, 0}, 40.0f, aspect);
  return sc;
}

// n random small triangles in a cube: the worst case for coherence and a stress test for the builder.
inline SceneAndCamera triangle_soup(uint32_t n, uint64_t seed, float aspect) {
  SceneAndCamera sc;
  Scene& s = sc.scene;
  const uint32_t m = s.add_material({0.6f, 0.6f, 0.6f});
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f);
  const float size = 0.5f * std::cbrt(8.0f / (float)std::max<uint32_t>(n, 1));
  for (uint32_t i = 0; i < n; ++i) {
    Vec3 c = {U(g), U(g), U(g)};
    s.add_tri(c, c + Vec3{U(g), U(g), U(g)} * size, c + Vec3{U(g), U(g), U(g)} * size, m);
  }
  s.env = {0.8f, 0.9f, 1.0f};
  s.finalize();
  sc.camera = Camera::look_at({0, 0, 3.2f}, {0, 0, 0}, {0, 1, 0}, 45.0f, aspect);
  return sc;
}

}  // namespace pt
