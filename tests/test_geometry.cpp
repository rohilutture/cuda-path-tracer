#include <cmath>
#include <vector>

#include "check.hpp"
#include "pt/bvh.h"
#include "pt/camera.h"
#include "pt/pathtrace.h"

using namespace pt;

TEST(tri_hit_miss_backface_parallel_and_range) {
  const Tri t = make_tri({0, 0, 0}, {1, 0, 0}, {0, 1, 0});  // in the z=0 plane
  float d;
  CHECK(intersect_tri(t, Ray{{0.25f, 0.25f, 1}, {0, 0, -1}, 1e-4f, kInf}, kInf, d));
  CHECK_NEAR(d, 1.0, 1e-6);
  CHECK(intersect_tri(t, Ray{{0.25f, 0.25f, -1}, {0, 0, 1}, 1e-4f, kInf}, kInf, d));   // from behind: double-sided
  CHECK(!intersect_tri(t, Ray{{0.8f, 0.8f, 1}, {0, 0, -1}, 1e-4f, kInf}, kInf, d));    // outside the triangle
  CHECK(!intersect_tri(t, Ray{{0.25f, 0.25f, 1}, {1, 0, 0}, 1e-4f, kInf}, kInf, d));   // parallel to the plane
  CHECK(!intersect_tri(t, Ray{{0.25f, 0.25f, 1}, {0, 0, 1}, 1e-4f, kInf}, kInf, d));   // pointing away
  CHECK(!intersect_tri(t, Ray{{0.25f, 0.25f, 1}, {0, 0, -1}, 1e-4f, 0.5f}, 0.5f, d));  // beyond tmax
  CHECK(!intersect_tri(t, Ray{{0.25f, 0.25f, 1e-5f}, {0, 0, -1}, 1e-4f, kInf}, kInf, d));  // inside tmin (self-hit guard)
  CHECK(!intersect_tri(make_tri({0, 0, 0}, {1, 0, 0}, {2, 0, 0}), Ray{{0.5f, 0, 1}, {0, 0, -1}, 1e-4f, kInf}, kInf, d));  // degenerate
}

TEST(shared_edge_has_no_cracks) {
  // a unit quad split along its diagonal: no ray through the quad may slip through the seam
  const Tri a = make_tri({0, 0, 0}, {1, 0, 0}, {1, 1, 0}), b = make_tri({0, 0, 0}, {1, 1, 0}, {0, 1, 0});
  Rng rng(7, 1);
  int misses = 0;
  for (int i = 0; i < 200000; ++i) {
    float x = rng.next_f(), y = rng.next_f();
    if (i % 4 == 0) y = x;  // aim directly at the seam
    Ray r{{x, y, 1}, {0, 0, -1}, 1e-4f, kInf};
    float d;
    misses += !(intersect_tri(a, r, kInf, d) || intersect_tri(b, r, kInf, d));
  }
  CHECK_MSG(misses == 0, std::to_string(misses) + " rays slipped through the seam");
}

TEST(box_slab_handles_axis_parallel_rays_on_slab_planes) {
  BVHNode n{{0, 0, 0}, 0, {1, 1, 1}, 0};
  auto hit = [&](Vec3 o, Vec3 d) { return hit_box(n, o, safe_inv_dir(d), 0.0f, kInf); };
  CHECK(hit({-1, 0.5f, 0.5f}, {1, 0, 0}));       // dir.y = dir.z = 0: no infinities may poison the result
  CHECK(hit({-1, 0.0f, 0.5f}, {1, 0, 0}));       // origin exactly on the y = 0 slab plane (regression: used to miss)
  CHECK(hit({-1, 0.0f, 0.0f}, {1, 0, 0}));       // exactly on the lower edge
  CHECK(hit({-1, 1.0f, 1.0f}, {1, -0.0f, -0.0f}));  // upper edge with negative-zero direction (boundary side is a sign convention)
  CHECK(!hit({-1, 2.0f, 0.5f}, {1, 0, 0}));      // parallel and outside
  CHECK(hit({0.5f, 0.5f, 0.5f}, {0, 1, 0}));     // origin inside
  CHECK(!hit({2, 0.5f, 0.5f}, {1, 0, 0}));       // box behind the ray
}

TEST(rng_is_uniform_reproducible_and_streams_differ) {
  Rng a(1, 5), b(1, 5), c(1, 6);
  double sum = 0, sq = 0;
  const int n = 400000;
  bool same = true, differ = false;
  for (int i = 0; i < n; ++i) {
    float x = a.next_f();
    CHECK(x >= 0.0f && x < 1.0f);
    sum += x; sq += (double)x * x;
    same &= (x == b.next_f());
    differ |= (x != c.next_f());
  }
  CHECK(same && differ);
  CHECK_NEAR(sum / n, 0.5, 0.003);
  CHECK_NEAR(sq / n - (sum / n) * (sum / n), 1.0 / 12.0, 0.002);
}

TEST(basis_is_orthonormal_for_all_directions) {
  Rng rng(3, 3);
  for (int i = 0; i < 20000; ++i) {
    const float z = 1 - 2 * rng.next_f(), r = std::sqrt(std::max(0.f, 1 - z * z)), p = 2 * kPi * rng.next_f();
    Vec3 n = {r * std::cos(p), r * std::sin(p), z}, t, b;
    if (i % 100 == 0) n = {0, 0, (i % 200 == 0) ? 1.f : -1.f};  // the poles are the classic failure case
    basis(n, t, b);
    CHECK_NEAR(length(t), 1, 1e-4); CHECK_NEAR(length(b), 1, 1e-4);
    CHECK_NEAR(dot(t, b), 0, 1e-4); CHECK_NEAR(dot(t, n), 0, 1e-4); CHECK_NEAR(dot(b, n), 0, 1e-4);
  }
}

TEST(cosine_sampling_has_the_right_moments) {
  Rng rng(11, 2);
  const Vec3 n = normalize({0.3f, -0.5f, 0.8f});
  Vec3 t, b;
  basis(n, t, b);
  const int N = 400000;
  double ez = 0, ez2 = 0, ex = 0, ex2 = 0;
  for (int i = 0; i < N; ++i) {
    const float u1 = rng.next_f();
    const float u2 = rng.next_f();
    const Vec3 d = sample_cosine(n, u1, u2);
    const double z = dot(d, n), x = dot(d, t);
    CHECK(z >= -1e-5);  // never below the surface
    ez += z; ez2 += z * z; ex += x; ex2 += x * x;
  }
  CHECK_NEAR(ez / N, 2.0 / 3.0, 0.003);   // E[cos] under pdf cos/pi
  CHECK_NEAR(ez2 / N, 0.5, 0.003);        // E[cos^2]
  CHECK_NEAR(ex / N, 0.0, 0.003);         // azimuthally symmetric
  CHECK_NEAR(ex2 / N, 0.25, 0.003);       // E[sin^2 cos^2 phi]
}

TEST(camera_rays_point_where_they_should) {
  const Camera c = Camera::look_at({0, 0, 5}, {0, 0, 0}, {0, 1, 0}, 90.0f, 2.0f);
  const Ray mid = c.ray(0.5f, 0.5f);
  CHECK_NEAR(mid.d.x, 0, 1e-6); CHECK_NEAR(mid.d.y, 0, 1e-6); CHECK_NEAR(mid.d.z, -1, 1e-6);
  // vfov 90: the top edge of the image is 45 degrees up; aspect 2: the right edge is atan(2) to the right
  const Ray top = c.ray(0.5f, 1.0f), right = c.ray(1.0f, 0.5f);
  CHECK_NEAR(std::atan2(top.d.y, -top.d.z) * 180 / kPi, 45.0, 1e-3);
  CHECK_NEAR(std::atan2(right.d.x, -right.d.z) * 180 / kPi, std::atan(2.0) * 180 / kPi, 1e-3);
  CHECK(right.d.x > 0 && c.ray(0.0f, 0.5f).d.x < 0);  // +u is image-right
}
