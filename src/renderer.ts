/**
 * WebGL2 GPU 蒙皮渲染器。
 * - uBoneMatrices uniform 数组注入
 * - 启动时校验 MAX_VERTEX_UNIFORM_VECTORS,骨骼数超限时自动分批渲染并警告,不崩溃
 */
import { Mat4 } from './math';
import { Skeleton } from './skeleton';

/** 为其他 uniform(uViewProj、uLightDir 等)预留的 vector 数 */
const RESERVED_VECTORS = 8;

const VERT_SRC = (maxBones: number) => `#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in uvec2 aBoneIndices; // 最多 2 个影响骨骼(整数属性)
layout(location=3) in vec2 aBoneWeights;

uniform mat4 uViewProj;
uniform mat4 uBoneMatrices[${maxBones}];

out vec3 vNormal;
out vec3 vWorldPos;

void main() {
  mat4 skin = uBoneMatrices[aBoneIndices.x] * aBoneWeights.x
            + uBoneMatrices[aBoneIndices.y] * aBoneWeights.y;
  vec4 world = skin * vec4(aPosition, 1.0);
  vWorldPos = world.xyz;
  vNormal = normalize(mat3(skin) * aNormal);
  gl_Position = uViewProj * world;
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vWorldPos;
uniform vec3 uLightDir;
out vec4 outColor;
void main() {
  float diff = max(dot(normalize(vNormal), normalize(uLightDir)), 0.0);
  vec3 base = vec3(0.35, 0.75, 0.9);
  outColor = vec4(base * (0.25 + 0.75 * diff), 1.0);
}
`;

export interface SkinMesh {
  positions: Float32Array;
  normals: Float32Array;
  /** 每顶点 2 个骨骼索引 */
  boneIndices: Uint16Array;
  boneWeights: Float32Array;
  indices: Uint16Array;
}

/**
 * 生成沿骨骼链的圆管网格,顶点蒙皮到相邻两根骨骼。
 * 每段骨骼之间插 2 个环(骨骼处权重 1.0,中间环 0.5/0.5),
 * 保证任意三角形引用的骨骼数 <= 2,分批渲染在最小容量(2)下依然可行。
 */
export function buildChainMesh(skeleton: Skeleton, sides = 8, radius = 0.12): SkinMesh {
  const height = 0.5; // 与 BONE_SPACING 一致
  const rings = skeleton.boneCount * 2 - 1; // 每段 2 个环
  const positions: number[] = [];
  const normals: number[] = [];
  const boneIndices: number[] = [];
  const boneWeights: number[] = [];
  const indices: number[] = [];

  for (let r = 0; r < rings; r++) {
    const y = (r * height) / 2;
    const t = y / height; // 以骨骼间距为单位的位置
    const b0 = Math.min(Math.floor(t), skeleton.boneCount - 1);
    const b1 = Math.min(b0 + 1, skeleton.boneCount - 1);
    const w1 = t - Math.floor(t); // 整数处为 0 -> 单骨骼绑定
    for (let s = 0; s < sides; s++) {
      const a = (s / sides) * Math.PI * 2;
      const nx = Math.cos(a), nz = Math.sin(a);
      positions.push(nx * radius, y, nz * radius);
      normals.push(nx, 0, nz);
      boneIndices.push(b0, b1);
      boneWeights.push(1 - w1, w1);
    }
  }
  for (let r = 0; r < rings - 1; r++) {
    for (let s = 0; s < sides; s++) {
      const a = r * sides + s;
      const b = r * sides + ((s + 1) % sides);
      const c = (r + 1) * sides + s;
      const d = (r + 1) * sides + ((s + 1) % sides);
      indices.push(a, c, b, b, c, d);
    }
  }
  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    boneIndices: Uint16Array.from(boneIndices),
    boneWeights: Float32Array.from(boneWeights),
    indices: Uint16Array.from(indices),
  };
}

interface Batch {
  /** 本批用到的骨骼原始索引 */
  boneIds: number[];
  /** 原始骨骼索引 -> 批内索引 */
  remap: Map<number, number>;
  /** 本批三角形索引(指向网格顶点) */
  triangles: number[];
}

/**
 * 贪心分批:遍历三角形,维护当前批的骨骼并集;
 * 加入某三角形会超出每批骨骼容量时,结算当前批并开新批。
 * 保证每个三角形的全部影响骨骼都在同一批内,蒙皮结果与单批绘制完全一致。
 */
export function buildBatches(mesh: SkinMesh, maxBonesPerBatch: number): Batch[] {
  const batches: Batch[] = [];
  let cur: Batch = { boneIds: [], remap: new Map(), triangles: [] };

  const flush = () => {
    if (cur.triangles.length > 0) batches.push(cur);
    cur = { boneIds: [], remap: new Map(), triangles: [] };
  };

  for (let t = 0; t < mesh.indices.length; t += 3) {
    const tri = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]];
    const needed = new Set<number>();
    for (const v of tri) {
      for (let k = 0; k < 2; k++) {
        if (mesh.boneWeights[v * 2 + k] > 0) needed.add(mesh.boneIndices[v * 2 + k]);
      }
    }
    const unionSize = new Set([...cur.boneIds, ...needed]).size;
    if (unionSize > maxBonesPerBatch) {
      if (needed.size > maxBonesPerBatch) {
        throw new Error(`单个三角形引用 ${needed.size} 根骨骼,超过每批容量 ${maxBonesPerBatch},无法分批`);
      }
      flush();
    }
    for (const b of needed) {
      if (!cur.remap.has(b)) {
        cur.remap.set(b, cur.boneIds.length);
        cur.boneIds.push(b);
      }
    }
    cur.triangles.push(...tri);
  }
  flush();
  return batches;
}

// ---------- 相机 ----------

function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const m = new Float32Array(16);
  const f = 1 / Math.tan(fovY / 2);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

function lookAt(eye: [number, number, number], target: [number, number, number], up: [number, number, number]): Mat4 {
  const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = (v: number[]) => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  };
  const cross = (a: number[], b: number[]) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  const m = new Float32Array(16);
  m.set([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0]);
  m[12] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
  m[13] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
  m[14] = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);
  m[15] = 1;
  return m;
}

function mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}

// ---------- 渲染器 ----------

export interface RendererInfo {
  maxVertexUniformVectors: number;
  maxBonesPerBatch: number;
  batchCount: number;
  batched: boolean;
  warnings: string[];
}

export class SkinnedRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private uViewProj: WebGLUniformLocation;
  private uBoneMatrices: WebGLUniformLocation;
  private uLightDir: WebGLUniformLocation;
  private batches: { vao: WebGLVertexArrayObject; indexCount: number; boneIds: number[] }[] = [];
  readonly info: RendererInfo;
  /** debugGL 模式下收集的探针结果 */
  readonly glProbes: string[] = [];

  /** 暴露给调试/自测读像素用 */
  get glContext(): WebGL2RenderingContext {
    return this.gl;
  }

  /**
   * @param forceMaxVectors 调试用:强制按此 uniform 上限分批(模拟低端显卡)
   */
  constructor(canvas: HTMLCanvasElement, skeleton: Skeleton, mesh: SkinMesh, forceMaxVectors?: number) {
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('当前环境不支持 WebGL2');
    this.gl = gl;

    const hwLimit = gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS) as number;
    const limit = forceMaxVectors ?? hwLimit;
    const maxBonesPerBatch = Math.floor((limit - RESERVED_VECTORS) / 4);
    const warnings: string[] = [];

    if (maxBonesPerBatch < 2) {
      throw new Error(`MAX_VERTEX_UNIFORM_VECTORS=${limit} 过低,连 2 根骨骼都放不下,无法渲染`);
    }

    let batches: Batch[];
    if (skeleton.boneCount <= maxBonesPerBatch) {
      batches = buildBatches(mesh, skeleton.boneCount);
    } else {
      warnings.push(
        `骨骼数 ${skeleton.boneCount} 超过本批容量 ${maxBonesPerBatch}` +
        `(MAX_VERTEX_UNIFORM_VECTORS=${limit},预留 ${RESERVED_VECTORS} 个给其他 uniform),` +
        `已降级为分批渲染`,
      );
      console.warn('[SkinnedRenderer]', warnings[0]);
      batches = buildBatches(mesh, maxBonesPerBatch);
    }

    this.info = {
      maxVertexUniformVectors: limit,
      maxBonesPerBatch,
      batchCount: batches.length,
      batched: batches.length > 1,
      warnings,
    };

    this.program = this.compile(VERT_SRC(maxBonesPerBatch), FRAG_SRC);
    this.uViewProj = this.mustUniform('uViewProj');
    this.uBoneMatrices = this.mustUniform('uBoneMatrices[0]');
    this.uLightDir = this.mustUniform('uLightDir');

    // 每批一个 VAO:共享位置/法线/权重 buffer,骨骼索引按批重映射
    const posBuf = this.makeBuffer(mesh.positions);
    const nrmBuf = this.makeBuffer(mesh.normals);
    const wgtBuf = this.makeBuffer(mesh.boneWeights);

    for (const batch of batches) {
      const remapped = new Uint16Array(mesh.boneIndices.length);
      for (let i = 0; i < mesh.boneIndices.length; i++) {
        remapped[i] = batch.remap.get(mesh.boneIndices[i]) ?? 0;
      }
      const boneBuf = this.makeBuffer(remapped);
      const idxBuf = this.makeIndexBuffer(Uint16Array.from(batch.triangles));

      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      this.bindAttrib(0, posBuf, 3, gl.FLOAT);
      this.bindAttrib(1, nrmBuf, 3, gl.FLOAT);
      this.bindAttrib(2, boneBuf, 2, gl.UNSIGNED_SHORT, false);
      this.bindAttrib(3, wgtBuf, 2, gl.FLOAT);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
      gl.bindVertexArray(null);

      this.batches.push({ vao, indexCount: batch.triangles.length, boneIds: batch.boneIds });
    }

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
  }

  private compile(vsSrc: string, fsSrc: string): WebGLProgram {
    const gl = this.gl;
    const compileOne = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error(`着色器编译失败: ${gl.getShaderInfoLog(sh)}`);
      }
      return sh;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compileOne(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(prog, compileOne(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`着色器链接失败: ${gl.getProgramInfoLog(prog)}`);
    }
    return prog;
  }

  private mustUniform(name: string): WebGLUniformLocation {
    const loc = this.gl.getUniformLocation(this.program, name);
    if (!loc) throw new Error(`找不到 uniform: ${name}`);
    return loc;
  }

  private makeBuffer(data: ArrayBufferView): WebGLBuffer {
    const gl = this.gl;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return buf;
  }

  /** 索引 buffer:上传与绑定都走 ELEMENT_ARRAY_BUFFER(部分驱动拒绝跨 target 复用) */
  private makeIndexBuffer(data: Uint16Array): WebGLBuffer {
    const gl = this.gl;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return buf;
  }

  private bindAttrib(loc: number, buf: WebGLBuffer, size: number, type: number, normalize = false): void {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    if (type === gl.UNSIGNED_SHORT && !normalize) {
      gl.vertexAttribIPointer(loc, size, type, 0, 0); // 骨骼索引用整数属性
    } else {
      gl.vertexAttribPointer(loc, size, type, normalize, 0, 0);
    }
  }

  /**
   * 绘制一帧。
   * @param skinMatrices 每根骨骼的最终蒙皮矩阵(global * inverseBind),扁平化长度 = boneCount*16
   * @param debugGL 为 true 时第一个批次后逐步检查 GL 错误(调试用)
   */
  draw(skinMatrices: Float32Array, aspect: number, debugGL = false): void {
    const gl = this.gl;
    const probe = (tag: string) => {
      if (debugGL) {
        const e = gl.getError();
        this.glProbes.push(`${tag}: ${e}`);
      }
    };
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.08, 0.08, 0.15, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const proj = perspective(Math.PI / 4, aspect, 0.1, 100);
    const view = lookAt([3.2, 1.6, 3.2], [0, 1.2, 0], [0, 1, 0]);
    const viewProj = mul(proj, view);

    gl.useProgram(this.program);
    probe('useProgram');
    gl.uniformMatrix4fv(this.uViewProj, false, viewProj);
    probe('uniform uViewProj');
    gl.uniform3f(this.uLightDir, 0.5, 1.0, 0.8);
    probe('uniform uLightDir');

    for (const batch of this.batches) {
      // 只上传本批骨骼的矩阵,批内索引已在建批时重映射
      const slice = new Float32Array(batch.boneIds.length * 16);
      batch.boneIds.forEach((boneId, i) => {
        slice.set(skinMatrices.subarray(boneId * 16, boneId * 16 + 16), i * 16);
      });
      gl.uniformMatrix4fv(this.uBoneMatrices, false, slice);
      probe('uniform uBoneMatrices');
      gl.bindVertexArray(batch.vao);
      probe('bindVertexArray');
      gl.drawElements(gl.TRIANGLES, batch.indexCount, gl.UNSIGNED_SHORT, 0);
      probe(`drawElements(${batch.indexCount})`);
    }
    gl.bindVertexArray(null);
  }
}
