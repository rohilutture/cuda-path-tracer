#pragma once
// Minimal test harness (no external dependencies).
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct TestCase { const char* name; std::function<void()> fn; };
inline std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }
inline int& failures() { static int f = 0; return f; }
struct Registrar { Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };

#define TEST(name) \
  static void test_##name(); \
  static Registrar reg_##name(#name, test_##name); \
  static void test_##name()

#define CHECK(c) \
  do { if (!(c)) { std::fprintf(stderr, "  CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures(); } } while (0)
#define CHECK_MSG(c, msg) \
  do { if (!(c)) { std::fprintf(stderr, "  CHECK failed %s:%d: %s [%s]\n", __FILE__, __LINE__, #c, std::string(msg).c_str()); ++failures(); } } while (0)
#define CHECK_NEAR(a, b, tol) \
  do { double va_ = (a), vb_ = (b); if (!(std::fabs(va_ - vb_) <= (tol))) { \
    std::fprintf(stderr, "  CHECK_NEAR failed %s:%d: %s ~ %s (%.6g vs %.6g, tol %.3g)\n", __FILE__, __LINE__, #a, #b, va_, vb_, (double)(tol)); \
    ++failures(); } } while (0)
