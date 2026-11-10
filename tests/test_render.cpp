#include <cmath>
#include <vector>

#include "check.hpp"
#include "pt/image.h"
#include "pt/render_cpu.h"
#include "pt/scenes.h"

using namespace pt;

namespace {

// Mean radiance over `n` paths starting at the origin in uniformly random directions.
double mean_radiance_from_center(const Scene& s, int n, int max_depth, uint64_t seed) {
  Rng rng(seed, 1);
  const SceneView v = s.view();
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    const float z = 1 - 2 * rng.next_f(), r = std::sqrt(std::max(0.f, 1 - z * z)), p = 2 * kPi * rng.next_f();
    sum += trace(v, Ray{{0, 0, 0}, {r * std::cos(p), r * std::sin(p), z}, 1e-4f, kInf}, rng, max_depth).y;
  }
  return sum / n;
}

Scene closed_sphere(float albedo, float emission) {
  Scene s;
  const uint32_t m = s.add_material({albedo, albedo, albedo}, {emission, emission, emission});
  add_icosphere(s, {0, 0, 0}, 1.0f, 3, m);
  s.finalize();
  return s;
}

double rmse(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  double se = 0;
  for (size_t i = 0; i < a.size(); ++i) { Vec3 d = a[i] - b[i]; se += dot(d, d); }
  return std::sqrt(se / (3.0 * a.size()));
}

}  // namespace

// Closed furnace: inside a sphere whose wall emits Le and reflects albedo rho, radiance is the same everywhere
// and in every direction, L = Le / (1 - rho) (L = Le + rho * L). An unbiased path tracer must converge to it.
TEST(furnace_interior_sphere_matches_closed_form) {
  struct C { float rho, le; };
  for (C c : {C{0.5f, 1.0f}, C{0.8f, 0.1f}, C{0.0f, 0.7f}, C{0.9f, 0.05f}}) {
    const double expect = c.le / (1.0 - c.rho);
    const double got = mean_radiance_from_center(closed_sphere(c.rho, c.le), 300000, 200, 17);
    CHECK_NEAR(got / expect, 1.0, 0.012);  // within 1.2 percent (several standard errors)
  }
}

TEST(furnace_depth_cap_biases_low_and_roulette_does_not) {
  // with max_depth = 2 only two bounces are counted: L = Le (1 + rho). Checks the loop's depth accounting.
  const double got = mean_radiance_from_center(closed_sphere(0.5f, 1.0f), 100000, 2, 5);
  CHECK_NEAR(got, 1.0 + 0.5, 0.01);
}

TEST(convex_white_diffuse_sphere_in_uniform_light_reflects_exactly_one) {
  Scene s;
  const uint32_t m = s.add_material({1, 1, 1});
  add_icosphere(s, {0, 0, 0}, 1.0f, 4, m);
  s.env = {1, 1, 1};
  s.finalize();
  Rng rng(1, 1);
  const SceneView v = s.view();
  for (int i = 0; i < 4000; ++i) {
    Ray r{{3, 0.3f * (rng.next_f() - 0.5f), 0.3f * (rng.next_f() - 0.5f)}, {-1, 0, 0}, 1e-4f, kInf};
    const Vec3 L = trace(v, r, rng, 8);
    CHECK_NEAR(L.x, 1.0, 1e-4);  // a convex albedo-1 surface can neither gain nor lose energy
  }
}

TEST(mirror_sphere_reflects_the_environment) {
  Scene s;
  const uint32_t m = s.add_material({1, 1, 1}, {0, 0, 0}, kMirror);
  add_icosphere(s, {0, 0, 0}, 1.0f, 3, m);
  s.env = {0.2f, 0.5f, 0.9f};
  s.finalize();
  Rng rng(2, 2);
  const Vec3 L = trace(s.view(), Ray{{3, 0.1f, 0.05f}, {-1, 0, 0}, 1e-4f, kInf}, rng, 8);
  CHECK_NEAR(L.x, 0.2, 1e-4); CHECK_NEAR(L.y, 0.5, 1e-4); CHECK_NEAR(L.z, 0.9, 1e-4);
}

TEST(emitter_seen_directly_and_absorbing_black_surface) {
  Scene s;
  const uint32_t light = s.add_material({0, 0, 0}, {5, 4, 3}), black = s.add_material({0, 0, 0});
  s.add_quad({-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, light);
  s.add_quad({-1, -1, -3}, {1, -1, -3}, {1, 1, -3}, {-1, 1, -3}, black);
  s.finalize();
  Rng rng(4, 4);
  const Vec3 L = trace(s.view(), Ray{{0, 0, 2}, {0, 0, -1}, 1e-4f, kInf}, rng, 8);
  CHECK_NEAR(L.x, 5, 1e-6); CHECK_NEAR(L.y, 4, 1e-6); CHECK_NEAR(L.z, 3, 1e-6);
  const Vec3 B = trace(s.view(), Ray{{0, 0, -1}, {0, 0, -1}, 1e-4f, kInf}, rng, 8);  // hits the black wall: nothing
  CHECK_NEAR(B.x + B.y + B.z, 0, 1e-9);
}

TEST(render_is_deterministic_and_thread_count_independent) {
  SceneAndCamera sc = cornell_box(1.0f, 2);
  const RenderSettings rs{48, 48, 4, 6, 7};
  const auto a = render_cpu(sc.scene.view(), sc.camera, rs, 1);
  const auto b = render_cpu(sc.scene.view(), sc.camera, rs, 4);
  const auto c = render_cpu(sc.scene.view(), sc.camera, rs, 4);
  bool same_ab = true, same_bc = true;
  for (size_t i = 0; i < a.size(); ++i) {
    same_ab &= (a[i].x == b[i].x && a[i].y == b[i].y && a[i].z == b[i].z);
    same_bc &= (b[i].x == c[i].x && b[i].y == c[i].y && b[i].z == c[i].z);
  }
  CHECK(same_ab);
  CHECK(same_bc);
  RenderSettings other = rs;
  other.seed = 8;
  CHECK(rmse(a, render_cpu(sc.scene.view(), sc.camera, other, 2)) > 1e-3);  // a different seed gives different noise
}

TEST(cornell_box_looks_like_a_cornell_box) {
  SceneAndCamera sc = cornell_box(1.0f, 2);
  const RenderSettings rs{64, 64, 96, 8, 1};
  const auto img = render_cpu(sc.scene.view(), sc.camera, rs, 4);
  double lum = 0, redL = 0, greenL = 0, rr = 0, rg = 0, gr = 0, gg = 0;
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x) {
      const Vec3 p = img[y * 64 + x];
      CHECK(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && p.x >= 0 && p.y >= 0 && p.z >= 0);
      lum += (p.x + p.y + p.z) / 3;
      if (x < 8 && y > 16 && y < 48) { redL += 1; rr += p.x; rg += p.y; }     // left wall strip
      if (x >= 56 && y > 16 && y < 48) { greenL += 1; gr += p.x; gg += p.y; }  // right wall strip
    }
  lum /= 64 * 64;
  CHECK(lum > 0.02 && lum < 3.0);
  CHECK(rr / redL > 1.5 * rg / redL);      // left wall is red: R well above G
  CHECK(gg / greenL > 1.3 * gr / greenL);  // right wall is green: G well above R
}

TEST(error_falls_as_one_over_sqrt_spp) {
  SceneAndCamera sc = cornell_box(1.0f, 2);
  auto at = [&](int spp, uint64_t seed) { return render_cpu(sc.scene.view(), sc.camera, {32, 32, spp, 6, seed}, 4); };
  const auto ref = at(1024, 99);
  const double e4 = rmse(at(4, 1), ref), e64 = rmse(at(64, 2), ref);
  // 16x the samples -> about 4x lower noise (a little less because the reference is itself noisy)
  CHECK_MSG(e4 / e64 > 2.8 && e4 / e64 < 5.5, "ratio " + std::to_string(e4 / e64));
}

TEST(png_writer_produces_a_valid_file) {
  std::vector<uint8_t> rgb(7 * 5 * 3);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = (uint8_t)(i * 7);
  const std::string path = "/tmp/pt_test.png";
  CHECK(write_png(path, 7, 5, rgb));
  FILE* f = std::fopen(path.c_str(), "rb");
  CHECK(f != nullptr);
  if (f) {
    uint8_t sig[8];
    CHECK(std::fread(sig, 1, 8, f) == 8 && sig[1] == 'P' && sig[2] == 'N' && sig[3] == 'G');
    std::fclose(f);
  }
}
