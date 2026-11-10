#pragma once
// Math primitives shared by host and device. Everything marked PT_HD compiles for both, so the CPU
// reference renderer runs exactly the same code as the GPU kernel.
#include <cfloat>
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define PT_HD __host__ __device__
#else
#define PT_HD
#endif

namespace pt {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInf = 3.0e38f;

struct Vec3 {
  float x, y, z;
  PT_HD float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

PT_HD inline Vec3 make3(float x, float y, float z) { return Vec3{x, y, z}; }
PT_HD inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
PT_HD inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
PT_HD inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
PT_HD inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
PT_HD inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
PT_HD inline Vec3 operator*(float s, Vec3 a) { return a * s; }
PT_HD inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
PT_HD inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
PT_HD inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
PT_HD inline float length(Vec3 a) { return sqrtf(dot(a, a)); }
PT_HD inline Vec3 normalize(Vec3 a) { return a * (1.0f / length(a)); }
PT_HD inline Vec3 vmin(Vec3 a, Vec3 b) { return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)}; }
PT_HD inline Vec3 vmax(Vec3 a, Vec3 b) { return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)}; }
PT_HD inline float max_component(Vec3 a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }

// PCG32: small, fast, good statistical quality, identical on host and device.
struct Rng {
  uint64_t state, inc;
  PT_HD Rng(uint64_t seed, uint64_t seq) : state(0), inc((seq << 1u) | 1u) {
    next_u32();
    state += seed;
    next_u32();
  }
  PT_HD uint32_t next_u32() {
    uint64_t old = state;
    state = old * 6364136223846793005ULL + inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
  }
  PT_HD float next_f() { return (float)(next_u32() >> 8) * (1.0f / 16777216.0f); }  // [0,1)
};

}  // namespace pt
