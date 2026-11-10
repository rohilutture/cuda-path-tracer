#pragma once
#include "geometry.h"

namespace pt {

struct Camera {
  Vec3 origin, lower_left, horizontal, vertical;

  static Camera look_at(Vec3 from, Vec3 at, Vec3 up, float vfov_deg, float aspect) {
    const float h = tanf(vfov_deg * kPi / 360.0f);
    const float vh = 2.0f * h, vw = aspect * vh;
    const Vec3 w = normalize(from - at), u = normalize(cross(up, w)), v = cross(w, u);
    Camera c;
    c.origin = from;
    c.horizontal = u * vw;
    c.vertical = v * vh;
    c.lower_left = from - c.horizontal * 0.5f - c.vertical * 0.5f - w;
    return c;
  }

  // (s, t) in [0,1]^2, t = 0 at the bottom of the image.
  PT_HD Ray ray(float s, float t) const {
    return {origin, normalize(lower_left + horizontal * s + vertical * t - origin), 1e-4f, kInf};
  }
};

}  // namespace pt
