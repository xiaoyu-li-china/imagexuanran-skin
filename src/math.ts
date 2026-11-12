/**
 * 手写数学库:列主序 mat4(Float32Array(16))与四元数。
 * 不依赖任何第三方引擎/动画库。
 */

export type Mat4 = Float32Array; // length 16, column-major
export type Quat = Float32Array; // length 4, [x,y,z,w]
export type Vec3 = Float32Array; // length 3

export function mat4Identity(out?: Mat4): Mat4 {
  const m = out ?? new Float32Array(16);
  m.fill(0);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** out = a * b(列主序,先应用 b 再应用 a) */
export function mat4Multiply(a: Mat4, b: Mat4, out?: Mat4): Mat4 {
  const o = out ?? new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

/** 由平移/旋转(四元数)/缩放合成矩阵 */
export function mat4FromTRS(t: Vec3, q: Quat, s: Vec3, out?: Mat4): Mat4 {
  const m = out ?? new Float32Array(16);
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const [sx, sy, sz] = s;

  m[0] = (1 - (yy + zz)) * sx;
  m[1] = (xy + wz) * sx;
  m[2] = (xz - wy) * sx;
  m[3] = 0;
  m[4] = (xy - wz) * sy;
  m[5] = (1 - (xx + zz)) * sy;
  m[6] = (yz + wx) * sy;
  m[7] = 0;
  m[8] = (xz + wy) * sz;
  m[9] = (yz - wx) * sz;
  m[10] = (1 - (xx + yy)) * sz;
  m[11] = 0;
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  m[15] = 1;
  return m;
}

/** 4x4 行列式(用于非奇异校验):沿第 4 列做余子式展开 */
export function mat4Determinant(m: Mat4): number {
  const a = (r: number, c: number) => m[c * 4 + r];
  const det3 = (
    a00: number, a01: number, a02: number,
    a10: number, a11: number, a12: number,
    a20: number, a21: number, a22: number,
  ) =>
    a00 * (a11 * a22 - a12 * a21) -
    a01 * (a10 * a22 - a12 * a20) +
    a02 * (a10 * a21 - a11 * a20);
  // 去掉第 skipRow 行、第 3 列后的 3x3 余子式
  const minor = (skipRow: number) => {
    const rows = [0, 1, 2, 3].filter((r) => r !== skipRow);
    return det3(
      a(rows[0], 0), a(rows[0], 1), a(rows[0], 2),
      a(rows[1], 0), a(rows[1], 1), a(rows[1], 2),
      a(rows[2], 0), a(rows[2], 1), a(rows[2], 2),
    );
  };
  return -a(0, 3) * minor(0) + a(1, 3) * minor(1) - a(2, 3) * minor(2) + a(3, 3) * minor(3);
}

/** 通用 4x4 求逆(伴随矩阵法);奇异时返回 null */
export function mat4Invert(m: Mat4, out?: Mat4): Mat4 | null {
  const o = out ?? new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return null;
  det = 1.0 / det;

  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return o;
}

/** 取矩阵平移分量(全局坐标) */
export function mat4GetTranslation(m: Mat4): [number, number, number] {
  return [m[12], m[13], m[14]];
}

// ---------- 四元数 ----------

export function quatNormalize(q: Quat, out?: Quat): Quat {
  const o = out ?? new Float32Array(4);
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (len < 1e-12) {
    o.set([0, 0, 0, 1]);
    return o;
  }
  o[0] = q[0] / len; o[1] = q[1] / len; o[2] = q[2] / len; o[3] = q[3] / len;
  return o;
}

export function quatFromAxisAngle(axis: [number, number, number], angleRad: number, out?: Quat): Quat {
  const o = out ?? new Float32Array(4);
  const half = angleRad / 2;
  const s = Math.sin(half);
  const len = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  o[0] = (axis[0] / len) * s;
  o[1] = (axis[1] / len) * s;
  o[2] = (axis[2] / len) * s;
  o[3] = Math.cos(half);
  return o;
}

/** 球面线性插值(自动处理最短路径) */
export function quatSlerp(a: Quat, b: Quat, t: number, out?: Quat): Quat {
  const o = out ?? new Float32Array(4);
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let cosHalf = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (cosHalf < 0) {
    cosHalf = -cosHalf;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  if (cosHalf > 0.9995) {
    // 近似线性,避免除零
    o[0] = a[0] + t * (bx - a[0]);
    o[1] = a[1] + t * (by - a[1]);
    o[2] = a[2] + t * (bz - a[2]);
    o[3] = a[3] + t * (bw - a[3]);
    return quatNormalize(o, o);
  }
  const half = Math.acos(cosHalf);
  const sinHalf = Math.sin(half);
  const ra = Math.sin((1 - t) * half) / sinHalf;
  const rb = Math.sin(t * half) / sinHalf;
  o[0] = a[0] * ra + bx * rb;
  o[1] = a[1] * ra + by * rb;
  o[2] = a[2] * ra + bz * rb;
  o[3] = a[3] * ra + bw * rb;
  return o;
}

export function vec3Lerp(a: Vec3, b: Vec3, t: number, out?: Vec3): Vec3 {
  const o = out ?? new Float32Array(3);
  o[0] = a[0] + (b[0] - a[0]) * t;
  o[1] = a[1] + (b[1] - a[1]) * t;
  o[2] = a[2] + (b[2] - a[2]) * t;
  return o;
}
