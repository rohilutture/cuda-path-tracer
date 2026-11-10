#pragma once
#include "common.h"

namespace pt {

struct Ray {
  Vec3 o, d;
  float tmin, tmax;
};

// Triangle stored as vertex + two edges (Moller-Trumbore needs the edges; saves recomputing them).
struct Tri {
  Vec3 v0, e1, e2;
};

PT_HD inline Tri make_tri(Vec3 a, Vec3 b, Vec3 c) { return Tri{a, b - a, c - a}; }
PT_HD inline Vec3 tri_normal(const Tri& t) { return normalize(cross(t.e1, t.e2)); }

// Double-sided Moller-Trumbore. Returns hit distance in `t` for hits with tmin < t < tmax.
PT_HD inline bool intersect_tri(const Tri& tri, const Ray& r, float tmax, float& t) {
  Vec3 p = cross(r.d, tri.e2);
  float det = dot(tri.e1, p);
  if (fabsf(det) < 1e-12f) return false;  // parallel (or degenerate triangle)
  float inv = 1.0f / det;
  Vec3 s = r.o - tri.v0;
  float u = dot(s, p) * inv;
  if (u < 0.0f || u > 1.0f) return false;
  Vec3 q = cross(s, tri.e1);
  float v = dot(r.d, q) * inv;
  if (v < 0.0f || u + v > 1.0f) return false;
  float tt = dot(tri.e2, q) * inv;
  if (tt <= r.tmin || tt >= tmax) return false;
  t = tt;
  return true;
}

struct Bounds {
  Vec3 lo, hi;
};
PT_HD inline Bounds empty_bounds() { return {{kInf, kInf, kInf}, {-kInf, -kInf, -kInf}}; }
PT_HD inline Bounds merge(Bounds a, Bounds b) { return {vmin(a.lo, b.lo), vmax(a.hi, b.hi)}; }
PT_HD inline Bounds merge(Bounds a, Vec3 p) { return {vmin(a.lo, p), vmax(a.hi, p)}; }
PT_HD inline float surface_area(Bounds b) {
  Vec3 d = b.hi - b.lo;
  if (d.x < 0 || d.y < 0 || d.z < 0) return 0.0f;
  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}
PT_HD inline Bounds tri_bounds(const Tri& t) {
  Vec3 v1 = t.v0 + t.e1, v2 = t.v0 + t.e2;
  return {vmin(t.v0, vmin(v1, v2)), vmax(t.v0, vmax(v1, v2))};
}

}  // namespace pt
