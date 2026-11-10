#pragma once
// Bounding volume hierarchy: binned-SAH builder (host) and stack-based closest-hit traversal (host + device).
//
// Layout: depth-first, flattened. An interior node's left child is the next node (index + 1); `a` holds the
// right child's index. A leaf stores `a` = first triangle and count in b >> 2 (triangles are reordered by the
// builder so a leaf's triangles are contiguous). Interior nodes store the split axis in b & 3, which lets
// traversal visit the child nearer to the ray origin first (ordered traversal), shrinking tmax sooner.
// Node size is 32 bytes (two nodes per 64-byte cache line).
#include <algorithm>
#include <cstddef>
#include <vector>

#include "geometry.h"

namespace pt {

struct BVHNode {
  Vec3 lo;
  uint32_t a;  // leaf: first triangle | interior: right child index
  Vec3 hi;
  uint32_t b;  // leaf: count << 2 | interior: axis (count bits are 0)
};
static_assert(sizeof(BVHNode) == 32, "BVHNode must stay 32 bytes");

PT_HD inline bool is_leaf(const BVHNode& n) { return (n.b >> 2) != 0; }

struct Hit {
  float t;
  int tri;  // -1 = miss
};

constexpr int kMaxStack = 64;

// Reciprocal direction with components clamped away from zero. A zero component would give inf, and a ray
// starting exactly on a slab plane then yields 0 * inf = NaN for one slab distance and inf for the other;
// min/max drop the NaN but the inf poisons the interval, so the ray wrongly misses a box it grazes. Using a
// huge-but-finite reciprocal keeps every product finite (0 * 1e20 = 0).
PT_HD inline float safe_recip(float d) { return 1.0f / (fabsf(d) < 1e-20f ? copysignf(1e-20f, d) : d); }
PT_HD inline Vec3 safe_inv_dir(Vec3 d) { return {safe_recip(d.x), safe_recip(d.y), safe_recip(d.z)}; }

// Slab test against a node (inclusive of the box boundary).
PT_HD inline bool hit_box(const BVHNode& n, Vec3 o, Vec3 inv, float tmin, float tmax) {
  float tx0 = (n.lo.x - o.x) * inv.x, tx1 = (n.hi.x - o.x) * inv.x;
  float ty0 = (n.lo.y - o.y) * inv.y, ty1 = (n.hi.y - o.y) * inv.y;
  float tz0 = (n.lo.z - o.z) * inv.z, tz1 = (n.hi.z - o.z) * inv.z;
  float t0 = fmaxf(fmaxf(fminf(tx0, tx1), fminf(ty0, ty1)), fmaxf(fminf(tz0, tz1), tmin));
  float t1 = fminf(fminf(fmaxf(tx0, tx1), fmaxf(ty0, ty1)), fminf(fmaxf(tz0, tz1), tmax));
  return t0 <= t1;
}

// Closest hit. `nodes` may be null/empty (empty scene -> miss).
PT_HD inline bool bvh_intersect(const BVHNode* nodes, uint32_t num_nodes, const Tri* tris, const Ray& r, Hit& hit) {
  hit.t = r.tmax;
  hit.tri = -1;
  if (num_nodes == 0) return false;
  const Vec3 inv = safe_inv_dir(r.d);
  const bool neg[3] = {inv.x < 0.0f, inv.y < 0.0f, inv.z < 0.0f};
  uint32_t stack[kMaxStack];
  int sp = 0;
  uint32_t node = 0;
  for (;;) {
    const BVHNode& n = nodes[node];
    if (hit_box(n, r.o, inv, r.tmin, hit.t)) {
      if (is_leaf(n)) {
        const uint32_t count = n.b >> 2;
        for (uint32_t i = 0; i < count; ++i) {
          float t;
          if (intersect_tri(tris[n.a + i], r, hit.t, t)) {
            hit.t = t;
            hit.tri = (int)(n.a + i);
          }
        }
      } else {
        const int axis = (int)(n.b & 3u);
        if (neg[axis]) {  // ray travels toward -axis: the right child is nearer
          stack[sp++] = node + 1;
          node = n.a;
        } else {
          stack[sp++] = n.a;
          node = node + 1;
        }
        continue;
      }
    }
    if (sp == 0) break;
    node = stack[--sp];
  }
  return hit.tri >= 0;
}

// Brute-force closest hit: the oracle for tests and the baseline the BVH is measured against.
PT_HD inline bool brute_intersect(const Tri* tris, uint32_t n, const Ray& r, Hit& hit) {
  hit.t = r.tmax;
  hit.tri = -1;
  for (uint32_t i = 0; i < n; ++i) {
    float t;
    if (intersect_tri(tris[i], r, hit.t, t)) {
      hit.t = t;
      hit.tri = (int)i;
    }
  }
  return hit.tri >= 0;
}

// ------------------------------------------------------------------ builder (host only)
struct BuildStats {
  int max_depth = 0;
  uint32_t leaves = 0, interior = 0;
  double sah_cost = 0;  // expected cost per ray under the SAH model (lower is better)
};

class BVHBuilder {
 public:
  BVHBuilder(std::vector<Tri>& tris, std::vector<uint32_t>* extra_perm, int max_leaf = 4, int bins = 16)
      : tris_(tris), max_leaf_(max_leaf), bins_(bins) {
    const size_t n = tris.size();
    order_.resize(n);
    for (size_t i = 0; i < n; ++i) order_[i] = (uint32_t)i;
    bounds_.resize(n);
    centroid_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      bounds_[i] = tri_bounds(tris[i]);
      centroid_[i] = (bounds_[i].lo + bounds_[i].hi) * 0.5f;
    }
    if (n) {
      root_bounds_ = empty_bounds();
      for (const Bounds& b : bounds_) root_bounds_ = merge(root_bounds_, b);
      build(0, (uint32_t)n, 0);
    }
    // apply the permutation to the triangles (and to any parallel per-triangle array)
    std::vector<Tri> sorted(n);
    for (size_t i = 0; i < n; ++i) sorted[i] = tris[order_[i]];
    tris = std::move(sorted);
    if (extra_perm) {
      std::vector<uint32_t> e(n);
      for (size_t i = 0; i < n; ++i) e[i] = (*extra_perm)[order_[i]];
      *extra_perm = std::move(e);
    }
    if (n && stats_.sah_cost == 0) compute_sah();
  }

  std::vector<BVHNode> nodes;
  BuildStats stats() const { return stats_; }

 private:
  uint32_t build(uint32_t begin, uint32_t end, int depth) {
    const uint32_t me = (uint32_t)nodes.size();
    nodes.push_back({});
    stats_.max_depth = std::max(stats_.max_depth, depth);

    Bounds nb = empty_bounds(), cb = empty_bounds();
    for (uint32_t i = begin; i < end; ++i) {
      nb = merge(nb, bounds_[order_[i]]);
      cb = merge(cb, centroid_[order_[i]]);
    }
    const uint32_t n = end - begin;
    auto make_leaf = [&] {
      nodes[me] = {nb.lo, begin, nb.hi, n << 2};
      ++stats_.leaves;
      return me;
    };
    if (n <= (uint32_t)max_leaf_ || depth >= 48) return make_leaf();  // depth cap keeps the traversal stack safe

    // longest centroid axis
    Vec3 ext = cb.hi - cb.lo;
    int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
    uint32_t mid;
    if (ext[axis] <= 0.0f) {
      mid = begin + n / 2;  // all centroids coincide: no spatial split exists, so split by count
    } else {
      mid = sah_split(begin, end, axis, cb, nb);
      if (mid == begin || mid == end) {
        if (n <= 16) return make_leaf();
        mid = begin + n / 2;
        auto first = order_.begin() + begin, last = order_.begin() + end;
        std::nth_element(first, first + n / 2, last,
                         [&](uint32_t x, uint32_t y) { return centroid_[x][axis] < centroid_[y][axis]; });
      }
    }
    ++stats_.interior;
    const uint32_t left = build(begin, mid, depth + 1);
    (void)left;  // always me + 1 by construction
    const uint32_t right = build(mid, end, depth + 1);
    nodes[me] = {nb.lo, right, nb.hi, (uint32_t)axis};
    return me;
  }

  // Binned SAH. Returns the partition point (partitions order_[begin,end) in place), or begin/end if
  // splitting is not cheaper than a leaf.
  uint32_t sah_split(uint32_t begin, uint32_t end, int axis, const Bounds& cb, const Bounds& nb) {
    struct Bin { Bounds b = empty_bounds(); uint32_t n = 0; };
    std::vector<Bin> bin(bins_);
    const float lo = cb.lo[axis], scale = (float)bins_ / (cb.hi[axis] - lo);
    auto bin_of = [&](uint32_t t) { return std::min(bins_ - 1, (int)((centroid_[t][axis] - lo) * scale)); };
    for (uint32_t i = begin; i < end; ++i) {
      Bin& b = bin[bin_of(order_[i])];
      b.b = merge(b.b, bounds_[order_[i]]);
      ++b.n;
    }
    // sweep: cost of splitting after bin k
    std::vector<float> left_area(bins_ - 1), right_area(bins_ - 1);
    std::vector<uint32_t> left_n(bins_ - 1), right_n(bins_ - 1);
    Bounds acc = empty_bounds();
    uint32_t cnt = 0;
    for (int k = 0; k < bins_ - 1; ++k) { acc = merge(acc, bin[k].b); cnt += bin[k].n; left_area[k] = surface_area(acc); left_n[k] = cnt; }
    acc = empty_bounds(); cnt = 0;
    for (int k = bins_ - 1; k > 0; --k) { acc = merge(acc, bin[k].b); cnt += bin[k].n; right_area[k - 1] = surface_area(acc); right_n[k - 1] = cnt; }
    int best = -1;
    float best_cost = kInf;
    for (int k = 0; k < bins_ - 1; ++k) {
      if (left_n[k] == 0 || right_n[k] == 0) continue;
      float c = left_area[k] * left_n[k] + right_area[k] * right_n[k];
      if (c < best_cost) { best_cost = c; best = k; }
    }
    if (best < 0) return begin;
    const float parent = surface_area(nb);
    const uint32_t n = end - begin;
    // cost model: traversal 1, intersection 1 per triangle. Leaf costs n.
    if (parent > 0 && 1.0f + best_cost / parent >= (float)n && n <= (uint32_t)max_leaf_) return begin;
    auto it = std::partition(order_.begin() + begin, order_.begin() + end, [&](uint32_t t) { return bin_of(t) <= best; });
    return (uint32_t)(it - order_.begin());
  }

  void compute_sah() {
    const float root = surface_area(root_bounds_);
    double cost = 0;
    for (const BVHNode& n : nodes) {
      Bounds b{n.lo, n.hi};
      const double p = root > 0 ? surface_area(b) / root : 0;
      cost += p * (is_leaf(n) ? (double)(n.b >> 2) : 1.0);
    }
    stats_.sah_cost = cost;
  }

  std::vector<Tri>& tris_;
  int max_leaf_, bins_;
  std::vector<uint32_t> order_;
  std::vector<Bounds> bounds_;
  std::vector<Vec3> centroid_;
  Bounds root_bounds_ = empty_bounds();
  BuildStats stats_;
};

}  // namespace pt
