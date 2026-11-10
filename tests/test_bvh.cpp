#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"
#include "pt/scene.h"
#include "pt/scenes.h"

using namespace pt;

namespace {

std::vector<Tri> random_tris(size_t n, uint64_t seed, float size) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<float> U(-1, 1);
  std::vector<Tri> t;
  for (size_t i = 0; i < n; ++i) {
    Vec3 c = {U(g), U(g), U(g)};
    t.push_back(make_tri(c, c + Vec3{U(g), U(g), U(g)} * size, c + Vec3{U(g), U(g), U(g)} * size));
  }
  return t;
}

// Structural invariants: every triangle in exactly one leaf, parents bound their children, layout rules hold.
std::string validate(const std::vector<BVHNode>& nodes, const std::vector<Tri>& tris, int* depth_out = nullptr) {
  if (tris.empty()) return nodes.empty() ? "" : "nodes for empty scene";
  std::vector<int> seen(tris.size(), 0);
  int max_depth = 0;
  struct Item { uint32_t n; int depth; };
  std::vector<Item> st = {{0, 0}};
  const float eps = 1e-5f;
  while (!st.empty()) {
    auto [i, d] = st.back(); st.pop_back();
    max_depth = std::max(max_depth, d);
    if (i >= nodes.size()) return "child index out of range";
    const BVHNode& n = nodes[i];
    if (is_leaf(n)) {
      for (uint32_t k = 0; k < (n.b >> 2); ++k) {
        if (n.a + k >= tris.size()) return "leaf range out of bounds";
        ++seen[n.a + k];
        const Bounds tb = tri_bounds(tris[n.a + k]);
        if (tb.lo.x < n.lo.x - eps || tb.lo.y < n.lo.y - eps || tb.lo.z < n.lo.z - eps ||
            tb.hi.x > n.hi.x + eps || tb.hi.y > n.hi.y + eps || tb.hi.z > n.hi.z + eps) return "leaf bounds do not contain triangle";
      }
    } else {
      const uint32_t l = i + 1, r = n.a;
      if (r <= l || r >= nodes.size()) return "bad right child index";
      if ((n.b & 3u) > 2u) return "bad axis";
      for (uint32_t c : {l, r}) {
        const BVHNode& ch = nodes[c];
        if (ch.lo.x < n.lo.x - eps || ch.lo.y < n.lo.y - eps || ch.lo.z < n.lo.z - eps ||
            ch.hi.x > n.hi.x + eps || ch.hi.y > n.hi.y + eps || ch.hi.z > n.hi.z + eps) return "parent does not bound child";
        st.push_back({c, d + 1});
      }
    }
  }
  for (int c : seen) if (c != 1) return "triangle not in exactly one leaf";
  if (depth_out) *depth_out = max_depth;
  return "";
}

// Compare BVH against brute force on `nrays` random rays. Returns number of disagreements.
int compare(const Scene& s, int nrays, uint64_t seed, bool from_inside_too = true) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<float> U(-1, 1);
  int bad = 0;
  const SceneView v = s.view();
  for (int i = 0; i < nrays; ++i) {
    Vec3 o = {U(g) * 2, U(g) * 2, U(g) * 2};
    if (!from_inside_too) o = o * 3.0f;
    Vec3 d = normalize(Vec3{U(g), U(g), U(g)});
    if (i % 7 == 0) d = {(i % 3 == 0) ? 1.f : 0.f, (i % 3 == 1) ? -1.f : 0.f, (i % 3 == 2) ? 1.f : 0.f};  // axis-aligned
    Ray r{o, d, 1e-4f, kInf};
    Hit a, b;
    const bool ha = bvh_intersect(v.nodes, v.num_nodes, v.tris, r, a), hb = brute_intersect(v.tris, v.num_tris, r, b);
    if (ha != hb) { ++bad; continue; }
    if (ha && (std::fabs(a.t - b.t) > 1e-5f * std::max(1.0f, b.t))) ++bad;  // ties may pick different triangles at the same t
  }
  return bad;
}

Scene make_scene(std::vector<Tri> tris, int leaf) {
  Scene s;
  s.add_material({0.5f, 0.5f, 0.5f});
  for (const Tri& t : tris) { s.tris.push_back(t); s.mat.push_back(0); }
  s.finalize(leaf);
  return s;
}

}  // namespace

TEST(bvh_matches_brute_force_across_sizes_and_leaf_sizes) {
  for (size_t n : {1u, 2u, 3u, 5u, 50u, 2000u}) {
    for (int leaf : {1, 4, 8}) {
      Scene s = make_scene(random_tris(n, n * 31 + leaf, 0.4f), leaf);
      CHECK_MSG(validate(s.nodes, s.tris).empty(), "n=" + std::to_string(n) + " leaf=" + std::to_string(leaf) + ": " + validate(s.nodes, s.tris));
      const int bad = compare(s, 3000, n + leaf);
      CHECK_MSG(bad == 0, "n=" + std::to_string(n) + " leaf=" + std::to_string(leaf) + ": " + std::to_string(bad) + " disagreements");
    }
  }
}

TEST(bvh_handles_empty_scene) {
  Scene s;
  s.finalize();
  Hit h;
  CHECK(!bvh_intersect(s.nodes.data(), (uint32_t)s.nodes.size(), s.tris.data(), Ray{{0, 0, 0}, {0, 0, 1}, 1e-4f, kInf}, h));
  CHECK(s.nodes.empty());
}

TEST(bvh_degenerate_geometry_still_builds_and_is_correct) {
  {  // 200 identical triangles: no spatial split exists
    std::vector<Tri> t(200, make_tri({0, 0, 0}, {1, 0, 0}, {0, 1, 0}));
    Scene s = make_scene(t, 4);
    int depth = 0;
    CHECK_MSG(validate(s.nodes, s.tris, &depth).empty(), validate(s.nodes, s.tris));
    CHECK(depth < 20);
    Hit h;
    CHECK(bvh_intersect(s.nodes.data(), (uint32_t)s.nodes.size(), s.tris.data(), Ray{{0.2f, 0.2f, 1}, {0, 0, -1}, 1e-4f, kInf}, h));
  }
  {  // coincident centroids, different sizes (concentric triangles)
    std::vector<Tri> t;
    for (int i = 1; i <= 100; ++i) t.push_back(make_tri({-0.01f * i, -0.01f * i, 0}, {0.01f * i, -0.01f * i, 0}, {0, 0.02f * i, 0}));
    Scene s = make_scene(t, 4);
    CHECK_MSG(validate(s.nodes, s.tris).empty(), validate(s.nodes, s.tris));
    CHECK(compare(s, 2000, 5) == 0);
  }
  {  // flat scene: everything in one plane (zero extent on an axis)
    std::vector<Tri> t = random_tris(500, 9, 0.3f);
    for (Tri& x : t) { x.v0.z = 0; x.e1.z = 0; x.e2.z = 0; }
    Scene s = make_scene(t, 4);
    CHECK_MSG(validate(s.nodes, s.tris).empty(), validate(s.nodes, s.tris));
    CHECK(compare(s, 3000, 6) == 0);
  }
  {  // degenerate (zero-area) triangles mixed in
    std::vector<Tri> t = random_tris(300, 3, 0.4f);
    for (int i = 0; i < 300; i += 5) t[i] = make_tri({0.1f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.1f});
    Scene s = make_scene(t, 4);
    CHECK(compare(s, 3000, 8) == 0);
  }
}

TEST(bvh_depth_fits_the_traversal_stack_on_large_irregular_geometry) {
  SceneAndCamera sc = noisy_scene(6, 1.0f);  // ~82k triangles, bumpy
  int depth = 0;
  CHECK_MSG(validate(sc.scene.nodes, sc.scene.tris, &depth).empty(), validate(sc.scene.nodes, sc.scene.tris));
  CHECK_MSG(depth < kMaxStack - 4, "depth " + std::to_string(depth));
  CHECK(std::isfinite(sc.scene.build_stats.sah_cost) && sc.scene.build_stats.sah_cost > 1);
  CHECK(compare(sc.scene, 4000, 99) == 0);
}

TEST(bvh_reorders_materials_with_triangles) {
  // two materials; after the builder permutes triangles, each triangle must still carry its own material
  Scene s;
  const uint32_t red = s.add_material({1, 0, 0}), blue = s.add_material({0, 0, 1});
  std::vector<Tri> t = random_tris(400, 4, 0.2f);
  for (size_t i = 0; i < t.size(); ++i) {
    t[i].v0.x += (i % 2 ? 5.0f : -5.0f);  // material is a function of position: blue on the right, red on the left
    s.tris.push_back(t[i]);
    s.mat.push_back(i % 2 ? blue : red);
  }
  s.finalize(4);
  for (size_t i = 0; i < s.tris.size(); ++i) CHECK((s.tris[i].v0.x > 0) == (s.mat[i] == blue));
}

TEST(sah_builder_beats_a_single_leaf_and_icosphere_counts_are_right) {
  CHECK(icosphere_tris(0).size() == 20 && icosphere_tris(3).size() == 20 * 64);
  Scene s;
  s.add_material({1, 1, 1});
  add_icosphere(s, {0, 0, 0}, 1.0f, 4, 0);
  s.finalize(4);
  CHECK(s.tris.size() == 5120);
  // SAH cost of a good tree is far below testing every triangle (cost = N for a single leaf)
  CHECK_MSG(s.build_stats.sah_cost < 5120.0 / 20, std::to_string(s.build_stats.sah_cost));
}
