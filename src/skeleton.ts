/**
 * 骨骼层级树:JSON 解析、非法数据校验、正向运动学(FK)。
 */
import { Mat4, mat4Multiply, mat4Invert } from './math';

export interface BoneDefJSON {
  name: string;
  /** 父节点索引,-1 表示根节点 */
  parent: number;
  /** 局部变换矩阵,列主序 16 个数 */
  localMatrix: number[];
}

export interface SkeletonJSON {
  bones: BoneDefJSON[];
}

export interface Skeleton {
  boneCount: number;
  names: string[];
  parents: Int32Array;
  /** 绑定姿态(静止姿态)下每根骨骼的局部矩阵 */
  bindLocals: Mat4[];
  /** 绑定姿态下的全局矩阵 */
  bindGlobals: Mat4[];
  /** 逆绑定矩阵,蒙皮用:skinMatrix = global * inverseBind */
  inverseBinds: Mat4[];
  nameToIndex: Map<string, number>;
}

export class SkeletonParseError extends Error {}

function fail(msg: string): never {
  throw new SkeletonParseError(`骨骼数据非法: ${msg}`);
}

/**
 * 解析并校验骨骼 JSON。
 * 校验项:结构完整性、父索引范围、节点重名、循环引用、矩阵数值合法性。
 */
export function parseSkeleton(json: unknown): Skeleton {
  if (typeof json !== 'object' || json === null) fail('顶层必须是对象');
  const bones = (json as SkeletonJSON).bones;
  if (!Array.isArray(bones)) fail('缺少 bones 数组');
  if (bones.length === 0) fail('bones 数组为空');
  if (bones.length > 1024) fail(`骨骼数量 ${bones.length} 超过上限 1024`);

  const names: string[] = [];
  const parents = new Int32Array(bones.length);
  const bindLocals: Mat4[] = [];
  const nameToIndex = new Map<string, number>();

  bones.forEach((b, i) => {
    if (typeof b !== 'object' || b === null) fail(`bones[${i}] 不是对象`);
    const { name, parent, localMatrix } = b as BoneDefJSON;

    if (typeof name !== 'string' || name.length === 0) fail(`bones[${i}].name 缺失或为空`);
    if (nameToIndex.has(name)) fail(`节点重名: "${name}" 在 bones[${nameToIndex.get(name)}] 与 bones[${i}] 重复出现`);

    if (typeof parent !== 'number' || !Number.isInteger(parent)) fail(`bones[${i}]("${name}").parent 必须是整数`);
    if (parent < -1 || parent >= bones.length) fail(`bones[${i}]("${name}").parent=${parent} 超出范围 [-1, ${bones.length - 1}]`);
    if (parent === i) fail(`bones[${i}]("${name}") 的父节点是自己,构成循环引用`);

    if (!Array.isArray(localMatrix) || localMatrix.length !== 16) fail(`bones[${i}]("${name}").localMatrix 必须是 16 个数的数组`);
    if (!localMatrix.every((v) => typeof v === 'number' && Number.isFinite(v))) fail(`bones[${i}]("${name}").localMatrix 含非有限数值`);

    names.push(name);
    parents[i] = parent;
    bindLocals.push(Float32Array.from(localMatrix));
    nameToIndex.set(name, i);
  });

  // 循环引用检测:沿父链向上走,若回到起点即成环
  for (let i = 0; i < bones.length; i++) {
    const seen = new Set<number>([i]);
    let p = parents[i];
    while (p !== -1) {
      if (seen.has(p)) {
        fail(`检测到循环引用: bones[${i}]("${names[i]}") 的祖先链回到 bones[${p}]("${names[p]}")`);
      }
      seen.add(p);
      p = parents[p];
    }
  }

  // 至少一个根节点
  if (![...parents].includes(-1)) fail('没有根节点(没有任何 parent === -1 的骨骼)');

  const skeleton: Skeleton = {
    boneCount: bones.length,
    names,
    parents,
    bindLocals,
    bindGlobals: [],
    inverseBinds: [],
    nameToIndex,
  };
  skeleton.bindGlobals = computeGlobals(skeleton, bindLocals);
  skeleton.inverseBinds = skeleton.bindGlobals.map((g, i) => {
    const inv = mat4Invert(g);
    if (!inv) fail(`bones[${i}]("${names[i]}") 的绑定姿态全局矩阵奇异,无法求逆`);
    return inv;
  });
  return skeleton;
}

/**
 * 正向运动学:由每根骨骼的局部矩阵计算全局矩阵。
 * global[i] = global[parent[i]] * local[i](根节点 global = local)
 * 按索引顺序计算即可,因为父节点索引合法性已校验,但这里仍用拓扑序保证正确性。
 */
export function computeGlobals(skeleton: Skeleton, locals: Mat4[]): Mat4[] {
  if (locals.length !== skeleton.boneCount) {
    throw new Error(`locals 长度 ${locals.length} 与骨骼数 ${skeleton.boneCount} 不一致`);
  }
  const globals: Mat4[] = new Array(skeleton.boneCount);
  const done = new Uint8Array(skeleton.boneCount);
  // 拓扑序:反复扫描直到全部完成(数据已校验无环,必然收敛)
  let remaining = skeleton.boneCount;
  while (remaining > 0) {
    let progressed = false;
    for (let i = 0; i < skeleton.boneCount; i++) {
      if (done[i]) continue;
      const p = skeleton.parents[i];
      if (p === -1) {
        globals[i] = Float32Array.from(locals[i]);
      } else {
        if (!done[p]) continue;
        globals[i] = mat4Multiply(globals[p], locals[i]);
      }
      done[i] = 1;
      remaining--;
      progressed = true;
    }
    if (!progressed) throw new Error('FK 计算失败:骨骼图存在环(应在解析阶段被拦截)');
  }
  return globals;
}
