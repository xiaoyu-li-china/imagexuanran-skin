/**
 * 最小自测(纯逻辑,无 WebGL 依赖,浏览器与 Node 均可运行):
 * 1. 骨骼 JSON 解析:正常数据 + 重名/循环引用等非法数据必须抛出可读错误
 * 2. FK:全局矩阵 = 父全局 * 局部
 * 3. Crossfade:idle -> walk 切换,在 w=0.5 的中间插值帧输出指定骨骼全局坐标
 * 4. 插值矩阵非奇异校验(det != 0)
 */
import { parseSkeleton, computeGlobals, SkeletonParseError } from './skeleton';
import {
  AnimationStateMachine, allocTRS, bindPoseFallback, blendTRS, sampleClip,
} from './animation';
import { mat4Determinant, mat4GetTranslation, mat4Multiply, mat4FromTRS } from './math';
import { makeSkeletonJSON, makeIdleClip, makeWalkClip } from './demoData';

// Node 环境下直接运行自测;浏览器环境由 main.ts 调用 runSelfTest()
declare const process: { versions?: { node?: string }; exit(code: number): void } | undefined;

export interface SelfTestResult {
  lines: string[];
  passed: boolean;
}

export function runSelfTest(): SelfTestResult {
  const lines: string[] = [];
  let failures = 0;
  const log = (s: string) => lines.push(s);
  const check = (cond: boolean, label: string) => {
    lines.push(`${cond ? 'PASS' : 'FAIL'}  ${label}`);
    if (!cond) failures++;
  };

  // ---- 1. 非法数据校验 ----
  log('== 1. 骨骼数据校验 ==');
  const expectThrow = (label: string, json: unknown, keyword: string) => {
    try {
      parseSkeleton(json);
      check(false, `${label} 应当报错但没有`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      check(e instanceof SkeletonParseError && msg.includes(keyword), `${label} -> ${msg}`);
    }
  };
  const good = makeSkeletonJSON();
  expectThrow('节点重名', { bones: [...good.bones, { ...good.bones[1], name: 'bone0', parent: 0 }] }, '重名');
  expectThrow('循环引用(自引用)', { bones: [{ name: 'a', parent: 0, localMatrix: good.bones[0].localMatrix }] }, '循环');
  expectThrow(
    '循环引用(互相引用)',
    {
      bones: [
        { name: 'a', parent: 1, localMatrix: good.bones[0].localMatrix },
        { name: 'b', parent: 0, localMatrix: good.bones[0].localMatrix },
      ],
    },
    '循环',
  );
  expectThrow('父索引越界', { bones: [{ name: 'a', parent: 99, localMatrix: good.bones[0].localMatrix }] }, '超出范围');
  expectThrow('矩阵长度错误', { bones: [{ name: 'a', parent: -1, localMatrix: [1, 2, 3] }] }, '16 个数');

  // ---- 2. FK ----
  log('== 2. 正向运动学 ==');
  const skeleton = parseSkeleton(good);
  const globals = computeGlobals(skeleton, skeleton.bindLocals);
  const tip = mat4GetTranslation(globals[skeleton.boneCount - 1]);
  const expectedY = (skeleton.boneCount - 1) * 0.5;
  check(Math.abs(tip[1] - expectedY) < 1e-5, `末端骨骼全局 Y = ${tip[1].toFixed(4)}(期望 ${expectedY})`);

  // ---- 3. Crossfade 中间帧 ----
  log('== 3. Crossfade 插值 ==');
  const idle = makeIdleClip();
  const walk = makeWalkClip();
  const fade = 0.5; // 秒
  const asm = new AnimationStateMachine(fade);
  asm.registerClip('idle', idle);
  asm.registerClip('walk', walk);

  const fallback = bindPoseFallback(skeleton);
  // t=1.0s 时从 idle 切到 walk;在 t=1.25s 处权重应为 0.5
  const switchAt = 1.0;
  const idleTimeAtSwitch = 1.0;
  asm.switchTo('walk', switchAt, idleTimeAtSwitch);
  const evalTime = switchAt + fade / 2;
  const w = asm.blendWeight(evalTime);
  check(Math.abs(w - 0.5) < 1e-6, `t=${evalTime}s 时 crossfade 权重 w = ${w}(期望 0.5)`);

  const trs = asm.evaluate(evalTime, switchAt, fallback, skeleton.boneCount);
  const locals = trs.map((tr) => mat4FromTRS(tr.t, tr.q, tr.s));
  const blendedGlobals = computeGlobals(skeleton, locals);

  const targetBone = skeleton.boneCount - 1; // 末端骨骼 bone5
  const pos = mat4GetTranslation(blendedGlobals[targetBone]);
  log(
    `中间插值帧(t=${evalTime}s, w=0.5)骨骼 "${skeleton.names[targetBone]}" 全局坐标: ` +
    `(${pos[0].toFixed(4)}, ${pos[1].toFixed(4)}, ${pos[2].toFixed(4)})`,
  );
  check(Number.isFinite(pos[0]) && Number.isFinite(pos[1]) && Number.isFinite(pos[2]), '全局坐标为有限值');

  // 交叉验证:手动分别采样 idle/walk 再按 w=0.5 混合,结果应与状态机一致
  const idleTRS = allocTRS(skeleton.boneCount);
  sampleClip(idle, idleTimeAtSwitch + (evalTime - switchAt), fallback, idleTRS);
  const walkTRS = allocTRS(skeleton.boneCount);
  sampleClip(walk, evalTime - switchAt, fallback, walkTRS);
  const manual = allocTRS(skeleton.boneCount);
  blendTRS(idleTRS, walkTRS, 0.5, manual);
  const manualGlobals = computeGlobals(skeleton, manual.map((tr) => mat4FromTRS(tr.t, tr.q, tr.s)));
  const manualPos = mat4GetTranslation(manualGlobals[targetBone]);
  const drift = Math.hypot(pos[0] - manualPos[0], pos[1] - manualPos[1], pos[2] - manualPos[2]);
  check(drift < 1e-5, `状态机结果与手动混合结果一致(偏差 ${drift.toExponential(2)})`);

  // ---- 4. 非奇异校验 ----
  log('== 4. 插值矩阵非奇异校验 ==');
  let allNonSingular = true;
  for (let i = 0; i < skeleton.boneCount; i++) {
    const detLocal = mat4Determinant(locals[i]);
    const detGlobal = mat4Determinant(blendedGlobals[i]);
    if (!Number.isFinite(detLocal) || Math.abs(detLocal) < 1e-8) {
      log(`FAIL  bones[${i}] 插值局部矩阵奇异, det=${detLocal}`);
      allNonSingular = false;
    }
    if (!Number.isFinite(detGlobal) || Math.abs(detGlobal) < 1e-8) {
      log(`FAIL  bones[${i}] 插值全局矩阵奇异, det=${detGlobal}`);
      allNonSingular = false;
    }
  }
  check(allNonSingular, `全部 ${skeleton.boneCount} 根骨骼的插值局部/全局矩阵均非奇异`);

  // 蒙皮矩阵(global * inverseBind)也非奇异,GPU 端才不会把顶点压没
  let skinOK = true;
  for (let i = 0; i < skeleton.boneCount; i++) {
    const skin = mat4Multiply(blendedGlobals[i], skeleton.inverseBinds[i]);
    const det = mat4Determinant(skin);
    if (!Number.isFinite(det) || Math.abs(det) < 1e-8) skinOK = false;
  }
  check(skinOK, '全部蒙皮矩阵非奇异');

  // 采样边界:旋转+缩放同时存在的轨道在任意时刻采样不应产生 NaN
  let nanFree = true;
  for (let t = 0; t <= 60; t++) {
    const trsT = asm.evaluate(t / 10, switchAt, fallback, skeleton.boneCount);
    for (const tr of trsT) {
      if (![...tr.t, ...tr.q, ...tr.s].every(Number.isFinite)) nanFree = false;
    }
  }
  check(nanFree, '0~6s 连续采样无 NaN/Infinity');

  log(failures === 0 ? '== 自测全部通过 ==' : `== ${failures} 项失败 ==`);
  return { lines, passed: failures === 0 };
}

// Node 直接运行:npm run selftest
const isNode = typeof process !== 'undefined' && !!process?.versions?.node;
if (isNode) {
  const { lines, passed } = runSelfTest();
  for (const l of lines) console.log(l);
  process!.exit(passed ? 0 : 1);
}
