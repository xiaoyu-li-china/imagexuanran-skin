/**
 * 浏览器入口:装配骨骼解析、动画状态机、WebGL2 蒙皮渲染,并在页面上跑自测。
 */
import { parseSkeleton, computeGlobals } from './skeleton';
import { AnimationStateMachine, bindPoseFallback } from './animation';
import { mat4FromTRS, mat4Multiply } from './math';
import { makeSkeletonJSON, makeIdleClip, makeWalkClip } from './demoData';
import { SkinnedRenderer, buildChainMesh } from './renderer';
import { runSelfTest } from './selftest';

const logEl = document.getElementById('log')!;
const statusEl = document.getElementById('status')!;
const log = (s: string) => {
  console.log(s);
  logEl.textContent += s + '\n';
  logEl.scrollTop = logEl.scrollHeight; // 始终滚动到最新一行
};

// ---- 自测(与 `npm run selftest` 同一份逻辑) ----
const selftest = runSelfTest();
for (const l of selftest.lines) log(l);

// ---- 装配渲染管线 ----
try {
  const skeleton = parseSkeleton(makeSkeletonJSON());
  log(`骨骼解析成功: ${skeleton.boneCount} 根骨骼 [${skeleton.names.join(', ')}]`);

  const canvas = document.getElementById('glcanvas') as HTMLCanvasElement;
  const params = new URLSearchParams(location.search);
  const forceMax = params.has('forceMaxVectors') ? Number(params.get('forceMaxVectors')) : undefined;

  const mesh = buildChainMesh(skeleton);
  const renderer = new SkinnedRenderer(canvas, skeleton, mesh, forceMax);
  const info = renderer.info;
  log(
    `MAX_VERTEX_UNIFORM_VECTORS=${info.maxVertexUniformVectors}, ` +
    `每批最多 ${info.maxBonesPerBatch} 根骨骼, 实际 ${info.batchCount} 批` +
    (info.batched ? '(已降级为分批渲染)' : '(单批渲染)'),
  );
  for (const w of info.warnings) log(`WARNING: ${w}`);

  const asm = new AnimationStateMachine(params.has('fade') ? Number(params.get('fade')) : 0.4);
  asm.registerClip('idle', makeIdleClip());
  asm.registerClip('walk', makeWalkClip());
  const fallback = bindPoseFallback(skeleton);

  let stateStart = 0; // 当前状态开始播放的全局时刻
  document.getElementById('btn-idle')!.onclick = () => {
    asm.switchTo('idle', clock(), clipTime());
    stateStart = clock();
  };
  document.getElementById('btn-walk')!.onclick = () => {
    asm.switchTo('walk', clock(), clipTime());
    stateStart = clock();
  };

  const t0 = performance.now();
  const clock = () => (performance.now() - t0) / 1000;
  const clipTime = () => clock() - stateStart;

  // 调试/自测:?autoSwitch=walk@2 表示 t=2s 时自动切换到 walk
  const autoSwitch = params.get('autoSwitch');
  let autoSwitched = false;

  const locals = skeleton.bindLocals.map((m) => Float32Array.from(m));
  const skinFlat = new Float32Array(skeleton.boneCount * 16);
  let frameCount = 0;

  function frame(): void {
    const now = clock();
    if (!autoSwitched && autoSwitch) {
      const [st, at] = autoSwitch.split('@');
      if (now >= Number(at)) {
        asm.switchTo(st as 'idle' | 'walk', now, clipTime());
        stateStart = now;
        autoSwitched = true;
        log(`autoSwitch: t=${now.toFixed(2)}s 切换到 ${st}`);
      }
    }
    const trs = asm.evaluate(now, stateStart, fallback, skeleton.boneCount);
    for (let i = 0; i < skeleton.boneCount; i++) mat4FromTRS(trs[i].t, trs[i].q, trs[i].s, locals[i]);
    const globals = computeGlobals(skeleton, locals);
    for (let i = 0; i < skeleton.boneCount; i++) {
      const skin = mat4Multiply(globals[i], skeleton.inverseBinds[i]);
      skinFlat.set(skin, i * 16);
    }

    const w = asm.blendWeight(now);
    statusEl.textContent =
      `state=${asm.currentState}  t=${now.toFixed(2)}s  fade_w=${w.toFixed(2)}  batches=${info.batchCount}`;

    const dpr = Math.min(window.devicePixelRatio, 2);
    const wpx = canvas.clientWidth * dpr;
    const hpx = canvas.clientHeight * dpr;
    if (canvas.width !== wpx || canvas.height !== hpx) {
      canvas.width = wpx;
      canvas.height = hpx;
    }
    renderer.draw(skinFlat, canvas.width / canvas.height, params.has('debugGL') && frameCount === 0);
    if (params.has('debugGL') && frameCount === 0) {
      for (const p of renderer.glProbes) log(`GL-PROBE ${p}`);
    }

    // 调试:?debugPixels 时第 3 帧读回像素,确认真有东西画到画布上
    frameCount++;
    if (params.has('debugPixels') && frameCount === 3) {
      try {
        const gl = renderer.glContext;
        const errBefore = gl.getError(); // draw 路径累积的错误
        const px = new Uint8Array(canvas.width * canvas.height * 4);
        gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, px);
        const errAfter = gl.getError();
        let nonBg = 0;
        for (let i = 0; i < px.length; i += 4) {
          // 背景色 (0.08,0.08,0.15) -> (20,20,38)
          if (Math.abs(px[i] - 20) > 8 || Math.abs(px[i + 1] - 20) > 8 || Math.abs(px[i + 2] - 38) > 8) nonBg++;
        }
        log(`debugPixels: 非背景像素 ${nonBg} / ${canvas.width * canvas.height}, errDraw=${errBefore}, errRead=${errAfter}`);
      } catch (e) {
        log(`debugPixels 读取失败: ${e instanceof Error ? e.message : e}`);
      }
    }

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  log('渲染循环已启动');
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  log(`ERROR: ${msg}`);
  statusEl.textContent = `ERROR: ${msg}`;
}
