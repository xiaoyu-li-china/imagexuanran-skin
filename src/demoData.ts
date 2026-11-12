/**
 * 演示数据:6 骨骼链式骨架 + idle / walk 两个动画片段(含旋转与缩放关键帧)。
 * 浏览器端与 Node 自测共用同一份数据。
 */
import { AnimationClip, AnimationTrack } from './animation';
import { SkeletonJSON } from './skeleton';
import { mat4FromTRS, quatFromAxisAngle } from './math';

export const BONE_SPACING = 0.5;
export const BONE_COUNT = 6;

/** 生成链式骨架 JSON:bone0 为根,沿 +Y 逐层偏移 */
export function makeSkeletonJSON(): SkeletonJSON {
  const bones = [];
  for (let i = 0; i < BONE_COUNT; i++) {
    const t = new Float32Array([0, i === 0 ? 0 : BONE_SPACING, 0]);
    const q = new Float32Array([0, 0, 0, 1]);
    const s = new Float32Array([1, 1, 1]);
    bones.push({
      name: `bone${i}`,
      parent: i === 0 ? -1 : i - 1,
      localMatrix: Array.from(mat4FromTRS(t, q, s)),
    });
  }
  return { bones };
}

interface KeySpec {
  time: number;
  /** 绕 Z 轴旋转角度(弧度) */
  rotZ: number;
  /** 均匀缩放 */
  scale: number;
}

function makeTrack(boneIndex: number, keys: KeySpec[]): AnimationTrack {
  const n = keys.length;
  const times = new Float32Array(n);
  const translations = new Float32Array(n * 3);
  const rotations = new Float32Array(n * 4);
  const scales = new Float32Array(n * 3);
  const baseT: [number, number, number] = [0, boneIndex === 0 ? 0 : BONE_SPACING, 0];
  keys.forEach((k, i) => {
    times[i] = k.time;
    translations.set(baseT, i * 3);
    rotations.set(quatFromAxisAngle([0, 0, 1], k.rotZ), i * 4);
    scales.set([k.scale, k.scale, k.scale], i * 3);
  });
  return { boneIndex, times, translations, rotations, scales };
}

/** idle:2 秒循环,轻微摇摆 + 呼吸缩放 */
export function makeIdleClip(): AnimationClip {
  const A = 0.08; // 摆幅(弧度)
  return {
    name: 'idle',
    duration: 2,
    tracks: [1, 2, 3, 4, 5].map((bone, idx) =>
      makeTrack(bone, [
        { time: 0, rotZ: 0, scale: 1 },
        { time: 0.5, rotZ: A * (idx % 2 === 0 ? 1 : -1), scale: 1.02 },
        { time: 1.0, rotZ: 0, scale: 1 },
        { time: 1.5, rotZ: -A * (idx % 2 === 0 ? 1 : -1), scale: 0.98 },
        { time: 2.0, rotZ: 0, scale: 1 },
      ]),
    ),
  };
}

/** walk:1 秒循环,大幅摆动 + 明显缩放脉冲 */
export function makeWalkClip(): AnimationClip {
  const A = 0.45;
  return {
    name: 'walk',
    duration: 1,
    tracks: [1, 2, 3, 4, 5].map((bone, idx) =>
      makeTrack(bone, [
        { time: 0, rotZ: A * (idx % 2 === 0 ? 1 : -1), scale: 1 },
        { time: 0.25, rotZ: 0, scale: 1.1 },
        { time: 0.5, rotZ: -A * (idx % 2 === 0 ? 1 : -1), scale: 1 },
        { time: 0.75, rotZ: 0, scale: 0.9 },
        { time: 1.0, rotZ: A * (idx % 2 === 0 ? 1 : -1), scale: 1 },
      ]),
    ),
  };
}
