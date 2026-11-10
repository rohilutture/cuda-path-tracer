#pragma once
#include <chrono>
#include <vector>

#include "bvh.h"

namespace pt {

enum MatType : uint32_t { kDiffuse = 0, kMirror = 1 };

// A surface can both emit and reflect (needed for the closed-furnace energy test); lights are just emissive diffuse.
struct Material {
  Vec3 albedo;
  Vec3 emission;
  uint32_t type;
};

// Plain-old-data view of a finalized scene: the same struct is used on the host and (with device pointers) on the GPU.
struct SceneView {
  const BVHNode* nodes;
  const Tri* tris;
  const uint32_t* mat;       // material index per (reordered) triangle
  const Material* materials;
  uint32_t num_nodes;
  uint32_t num_tris;
  Vec3 env;                  // constant environment radiance seen by rays that escape
};

class Scene {
 public:
  std::vector<Tri> tris;
  std::vector<uint32_t> mat;
  std::vector<Material> materials;
  std::vector<BVHNode> nodes;
  Vec3 env{0, 0, 0};
  BuildStats build_stats;
  double build_ms = 0;

  uint32_t add_material(Vec3 albedo, Vec3 emission = {0, 0, 0}, MatType t = kDiffuse) {
    materials.push_back({albedo, emission, (uint32_t)t});
    return (uint32_t)materials.size() - 1;
  }
  void add_tri(Vec3 a, Vec3 b, Vec3 c, uint32_t m) {
    tris.push_back(make_tri(a, b, c));
    mat.push_back(m);
  }
  void add_quad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, uint32_t m) {  // a,b,c,d in order around the quad
    add_tri(a, b, c, m);
    add_tri(a, c, d, m);
  }

  // Build the BVH (reorders tris/mat). Call once after adding geometry.
  void finalize(int max_leaf = 4);

  SceneView view() const {
    return {nodes.data(), tris.data(), mat.data(), materials.data(), (uint32_t)nodes.size(), (uint32_t)tris.size(), env};
  }
};

inline void Scene::finalize(int max_leaf) {
  const auto t0 = std::chrono::steady_clock::now();
  BVHBuilder b(tris, &mat, max_leaf);
  nodes = std::move(b.nodes);
  build_stats = b.stats();
  build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

inline Bounds scene_bounds(const Scene& s) {
  Bounds b = empty_bounds();
  for (const Tri& t : s.tris) b = merge(b, tri_bounds(t));
  return b;
}

}  // namespace pt
