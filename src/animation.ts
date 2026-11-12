/**
 * 动画片段采样与最小状态机(idle / walk),支持秒级时间驱动的 Crossfade。
 */
import { Mat4, Quat, Vec3, mat4FromTRS, quatSlerp, vec3Lerp, quatNormalize } from './math';
import { Skeleton } from './skeleton';

/** 单根骨骼的一条动画轨道:关键帧时间 + TRS */
export interface AnimationTrack {
  boneIndex: number;
  times: Float32Array;        // 升序秒
  translations: Float32Array; // times.length * 3
  rotations: Float32Array;    // times.length * 4 (quat xyzw)
  scales: Float32Array;       // times.length * 3
}

export interface AnimationClip {
  name: string;
  /** 片段时长(秒) */
  duration: number;
  tracks: AnimationTrack[];
}

export interface TRS {
  t: Vec3;
  q: Quat;
  s: Vec3;
}

/** 在单条轨道上按时间采样(循环),线性插值 T/S,球面插值 R */
export function sampleTrack(track: AnimationTrack, timeSec: number, duration: number, out: TRS): TRS {
  const times = track.times;
  const n = times.length;
  const t = ((timeSec % duration) + duration) % duration;

  let i = 0;
  while (i < n - 1 && times[i + 1] < t) i++;
  const i0 = i;
  const i1 = Math.min(i + 1, n - 1);
  const span = times[i1] - times[i0];
  const alpha = span > 1e-9 ? Math.min(1, Math.max(0, (t - times[i0]) / span)) : 0;

  const read3 = (arr: Float32Array, idx: number): Vec3 =>
    new Float32Array([arr[idx * 3], arr[idx * 3 + 1], arr[idx * 3 + 2]]);
  const read4 = (arr: Float32Array, idx: number): Quat =>
    new Float32Array([arr[idx * 4], arr[idx * 4 + 1], arr[idx * 4 + 2], arr[idx * 4 + 3]]);

  vec3Lerp(read3(track.translations, i0), read3(track.translations, i1), alpha, out.t);
  quatSlerp(read4(track.rotations, i0), read4(track.rotations, i1), alpha, out.q);
  vec3Lerp(read3(track.scales, i0), read3(track.scales, i1), alpha, out.s);
  return out;
}

/**
 * 采样整个片段,输出每根骨骼的局部 TRS。
 * 没有轨道的骨骼回退到绑定姿态(从 bindLocal 矩阵分解代价高,这里直接传入回退值)。
 */
export function sampleClip(clip: AnimationClip, timeSec: number, fallback: TRS[], out: TRS[]): void {
  for (let i = 0; i < out.length; i++) {
    out[i].t.set(fallback[i].t);
    out[i].q.set(fallback[i].q);
    out[i].s.set(fallback[i].s);
  }
  for (const track of clip.tracks) {
    sampleTrack(track, timeSec, clip.duration, out[track.boneIndex]);
  }
}

/** 两组 TRS 按权重混合:w=0 全 a,w=1 全 b */
export function blendTRS(a: TRS[], b: TRS[], w: number, out: TRS[]): void {
  for (let i = 0; i < out.length; i++) {
    vec3Lerp(a[i].t, b[i].t, w, out[i].t);
    quatSlerp(a[i].q, b[i].q, w, out[i].q);
    vec3Lerp(a[i].s, b[i].s, w, out[i].s);
  }
}

export function trsToLocals(trs: TRS[], out: Mat4[]): void {
  for (let i = 0; i < trs.length; i++) {
    mat4FromTRS(trs[i].t, trs[i].q, trs[i].s, out[i]);
  }
}

export type AnimStateName = 'idle' | 'walk';

/**
 * 最小动画状态机。
 * 切换状态时记录旧状态与切换时刻,在 fadeDuration 秒内把权重从旧动画平滑过渡到新动画,
 * 权重由秒级时间驱动:w = clamp((now - switchTime) / fadeDuration)。
 */
export class AnimationStateMachine {
  private clips = new Map<AnimStateName, AnimationClip>();
  private current: AnimStateName = 'idle';
  private previous: AnimStateName | null = null;
  /** 切换发生的时刻(秒,由外部时钟提供) */
  private switchTimeSec = 0;
  /** 旧动画在切换时刻的本地时间,保证过渡期间旧动画从断点继续播 */
  private previousClipTimeSec = 0;
  readonly fadeDurationSec: number;

  constructor(fadeDurationSec = 0.3) {
    this.fadeDurationSec = fadeDurationSec;
  }

  registerClip(state: AnimStateName, clip: AnimationClip): void {
    this.clips.set(state, clip);
  }

  get currentState(): AnimStateName {
    return this.current;
  }

  /** 当前 crossfade 权重(1 = 完全在新动画上) */
  blendWeight(nowSec: number): number {
    if (this.previous === null) return 1;
    return Math.min(1, Math.max(0, (nowSec - this.switchTimeSec) / this.fadeDurationSec));
  }

  switchTo(state: AnimStateName, nowSec: number, currentClipTimeSec: number): void {
    if (state === this.current && this.previous === null) return;
    if (!this.clips.has(state)) throw new Error(`未注册的动画状态: ${state}`);
    this.previous = this.current;
    this.previousClipTimeSec = currentClipTimeSec;
    this.current = state;
    this.switchTimeSec = nowSec;
  }

  /**
   * 计算当前时刻每根骨骼的局部 TRS(已含 crossfade 混合)。
   * @param nowSec 全局时钟(秒)
   * @param stateStartSec 当前状态开始播放的全局时刻
   */
  evaluate(nowSec: number, stateStartSec: number, fallback: TRS[], boneCount: number): TRS[] {
    const clip = this.clips.get(this.current);
    if (!clip) throw new Error(`状态 ${this.current} 没有对应动画片段`);
    const curTime = nowSec - stateStartSec;

    const cur = allocTRS(boneCount);
    sampleClip(clip, curTime, fallback, cur);

    const w = this.blendWeight(nowSec);
    if (this.previous === null || w >= 1) return cur;

    const prevClip = this.clips.get(this.previous)!;
    const prev = allocTRS(boneCount);
    // 旧动画从切换时的断点继续走,过渡更自然
    sampleClip(prevClip, this.previousClipTimeSec + (nowSec - this.switchTimeSec), fallback, prev);

    const out = allocTRS(boneCount);
    blendTRS(prev, cur, w, out);
    return out;
  }
}

export function allocTRS(n: number): TRS[] {
  return Array.from({ length: n }, () => ({
    t: new Float32Array(3),
    q: quatNormalize(new Float32Array([0, 0, 0, 1])),
    s: new Float32Array([1, 1, 1]),
  }));
}

/** 从骨骼绑定姿态构造回退 TRS(假设局部矩阵只含 TRS,无剪切) */
export function bindPoseFallback(skeleton: Skeleton): TRS[] {
  return skeleton.bindLocals.map((m) => {
    const t = new Float32Array([m[12], m[13], m[14]]);
    const sx = Math.hypot(m[0], m[1], m[2]);
    const sy = Math.hypot(m[4], m[5], m[6]);
    const sz = Math.hypot(m[8], m[9], m[10]);
    // 由旋转部分提取四元数(先除掉缩放)
    const r00 = m[0] / sx, r01 = m[4] / sy, r02 = m[8] / sz;
    const r10 = m[1] / sx, r11 = m[5] / sy, r12 = m[9] / sz;
    const r20 = m[2] / sx, r21 = m[6] / sy, r22 = m[10] / sz;
    const trace = r00 + r11 + r22;
    const q = new Float32Array(4);
    if (trace > 0) {
      const s = Math.sqrt(trace + 1) * 2;
      q[3] = s / 4; q[0] = (r21 - r12) / s; q[1] = (r02 - r20) / s; q[2] = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
      const s = Math.sqrt(1 + r00 - r11 - r22) * 2;
      q[3] = (r21 - r12) / s; q[0] = s / 4; q[1] = (r01 + r10) / s; q[2] = (r02 + r20) / s;
    } else if (r11 > r22) {
      const s = Math.sqrt(1 + r11 - r00 - r22) * 2;
      q[3] = (r02 - r20) / s; q[0] = (r01 + r10) / s; q[1] = s / 4; q[2] = (r12 + r21) / s;
    } else {
      const s = Math.sqrt(1 + r22 - r00 - r11) * 2;
      q[3] = (r10 - r01) / s; q[0] = (r02 + r20) / s; q[1] = (r12 + r21) / s; q[2] = s / 4;
    }
    return { t, q: quatNormalize(q), s: new Float32Array([sx, sy, sz]) };
  });
}
