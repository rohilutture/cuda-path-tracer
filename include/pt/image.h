#pragma once
// Tone mapping and a dependency-free PNG writer (uncompressed "stored" deflate blocks: big files, zero libs).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"

namespace pt {

inline uint32_t crc32_update(uint32_t crc, const uint8_t* p, size_t n) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    init = true;
  }
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
  return ~crc;
}

inline void put_be32(std::vector<uint8_t>& v, uint32_t x) {
  for (int s = 24; s >= 0; s -= 8) v.push_back((uint8_t)(x >> s));
}

inline void png_chunk(FILE* f, const char* type, const std::vector<uint8_t>& data) {
  std::vector<uint8_t> hdr;
  put_be32(hdr, (uint32_t)data.size());
  std::fwrite(hdr.data(), 1, 4, f);
  std::vector<uint8_t> body(type, type + 4);
  body.insert(body.end(), data.begin(), data.end());
  std::fwrite(body.data(), 1, body.size(), f);
  std::vector<uint8_t> crc;
  put_be32(crc, crc32_update(0, body.data(), body.size()));
  std::fwrite(crc.data(), 1, 4, f);
}

// rgb: width*height*3 bytes.
inline bool write_png(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  std::fwrite(sig, 1, 8, f);
  std::vector<uint8_t> ihdr;
  put_be32(ihdr, (uint32_t)w); put_be32(ihdr, (uint32_t)h);
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8-bit RGB
  png_chunk(f, "IHDR", ihdr);
  std::vector<uint8_t> raw;  // filter byte 0 per scanline
  raw.reserve((size_t)h * (w * 3 + 1));
  for (int y = 0; y < h; ++y) {
    raw.push_back(0);
    raw.insert(raw.end(), rgb.begin() + (size_t)y * w * 3, rgb.begin() + (size_t)(y + 1) * w * 3);
  }
  std::vector<uint8_t> z = {0x78, 0x01};  // zlib header
  uint32_t a = 1, b = 0;
  for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
  for (size_t pos = 0; pos < raw.size() || pos == 0;) {
    size_t n = std::min<size_t>(65535, raw.size() - pos);
    z.push_back(pos + n >= raw.size() ? 1 : 0);  // BFINAL
    z.push_back(n & 0xff); z.push_back(n >> 8); z.push_back(~n & 0xff); z.push_back((~n >> 8) & 0xff);
    z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
    pos += n;
    if (raw.empty()) break;
  }
  put_be32(z, (b << 16) | a);
  png_chunk(f, "IDAT", z);
  png_chunk(f, "IEND", {});
  std::fclose(f);
  return true;
}

// Linear radiance -> 8-bit sRGB-ish (clamp + gamma 2.2).
inline std::vector<uint8_t> tonemap(const std::vector<Vec3>& img, float exposure = 1.0f) {
  std::vector<uint8_t> out(img.size() * 3);
  for (size_t i = 0; i < img.size(); ++i) {
    const float c[3] = {img[i].x, img[i].y, img[i].z};
    for (int k = 0; k < 3; ++k) {
      float v = std::pow(std::min(1.0f, std::max(0.0f, c[k] * exposure)), 1.0f / 2.2f);
      out[i * 3 + k] = (uint8_t)(v * 255.0f + 0.5f);
    }
  }
  return out;
}

}  // namespace pt
