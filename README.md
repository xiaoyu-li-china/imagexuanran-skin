# imagexuanran-skin — WebGL2 骨骼动画基础管线

原生 **WebGL2 + GLSL + TypeScript + Vite** 实现的骨骼动画最小管线,不依赖 Three.js / Babylon.js 等任何引擎或动画库。

## 运行

```bash
npm install
npm run dev        # 开发服务器
npm run selftest   # Node 端纯逻辑自测(骨骼解析/FK/Crossfade/非奇异校验)
npm run typecheck  # TypeScript 检查
npm run build      # 产物构建
```

浏览器打开页面后:自测结果自动输出到页面日志与控制台;`Idle` / `Walk` 按钮切换动画(带 Crossfade)。

### 调试 URL 参数

| 参数 | 作用 |
|---|---|
| `?forceMaxVectors=16` | 模拟低端显卡的 uniform 上限,验证分批渲染降级 |
| `?autoSwitch=walk@2` | t=2s 时自动切换到 walk(无头验证用) |
| `?fade=2` | 设置 Crossfade 时长(秒) |
| `?debugPixels=1` | 第 3 帧 readPixels 统计非背景像素,确认真实渲染 |
| `?debugGL=1` | 首帧逐步 GL 错误探针 |

## 骨骼矩阵层级计算(FK)

骨骼层级用 JSON 描述(`src/demoData.ts` 有示例):

```json
{ "bones": [ { "name": "bone0", "parent": -1, "localMatrix": [/* 列主序 16 数 */] } ] }
```

- **解析与校验**(`src/skeleton.ts` `parseSkeleton`):检查结构完整性、父索引范围、**节点重名**、**循环引用**(自引用 / 祖先链成环)、矩阵数值合法性,非法数据抛出带具体骨骼名与索引的 `SkeletonParseError`。
- **正向运动学**(`computeGlobals`):按拓扑序递推

  ```
  global[root] = local[root]
  global[i]    = global[parent[i]] * local[i]     (列主序,列向量约定)
  ```
- **蒙皮矩阵**:解析时对绑定姿态全局矩阵求逆得到 `inverseBind`(奇异矩阵会被拒绝),运行时

  ```
  skinMatrix[i] = global[i] * inverseBind[i]
  ```

  每帧把全部 `skinMatrix` 扁平化后通过 `uniformMatrix4fv` 注入顶点着色器的 `uBoneMatrices` 数组,GPU 端按顶点权重加权求和:

  ```glsl
  mat4 skin = uBoneMatrices[idx0] * w0 + uBoneMatrices[idx1] * w1;
  ```

## Crossfade 算法原理

状态机(`src/animation.ts` `AnimationStateMachine`)持有 `idle` / `walk` 两个片段。切换时**不跳变**,做法:

1. 记录旧状态、旧动画在切换时刻的本地时间(断点)、切换发生的全局时刻 `t_switch`;
2. 过渡期内每帧同时采样新旧两个片段——旧动画从断点继续播放,避免姿态回跳;
3. 权重由**秒级时间**驱动:`w = clamp((t_now - t_switch) / fadeDuration, 0, 1)`;
4. 按分量混合每根骨骼的局部 TRS:平移/缩放 **lerp**,旋转用四元数 **slerp**(自动取最短路径,近零夹角退化为归一化 lerp 防除零);
5. `w` 到达 1 后旧状态释放,只采样新动画。

注意:混合发生在**局部 TRS 空间**而非矩阵空间——逐元素插值矩阵会破坏旋转的正交性,而 TRS 插值保证结果始终是合法刚体变换 + 缩放。

## 分批渲染限制

GPU 蒙皮需要把全部骨骼矩阵放进 `uBoneMatrices` uniform 数组,但顶点着色器的 uniform 容量有硬件上限(`MAX_VERTEX_UNIFORM_VECTORS`,一个 mat4 占 4 个 vector):

```
每批骨骼容量 = floor((MAX_VERTEX_UNIFORM_VECTORS - 预留 8) / 4)
```

- 骨骼数 ≤ 容量:单批渲染;
- 骨骼数 > 容量:**自动降级为分批渲染**并 `console.warn` 警告(页面日志同步显示),不崩溃。分批算法(`buildBatches`)贪心遍历三角形,维护当前批的骨骼并集,超容量即结算开新批,保证每个三角形的全部影响骨骼落在同一批内;每批有独立 VAO(骨骼索引重映射到批内 0..k-1),绘制时只上传该批的矩阵切片;
- 极端情况(单个三角形引用的骨骼数就超过容量)无法分批,抛出可读错误而非静默渲染错误。演示网格在生成时保证每个三角形最多引用 2 根骨骼,使最小容量(2)下依然可分批。

## 最小自测

`npm run selftest`(浏览器端同一份逻辑自动运行)覆盖:

1. 非法数据:重名、自引用/互相引用环、父索引越界、矩阵长度错误 → 必须抛出可读错误;
2. FK:6 骨骼链末端全局坐标 = 2.5;
3. Crossfade:t=1.0s 切换、fade=0.5s,在 t=1.25s(w=0.5)的中间插值帧输出末端骨骼 `bone5` 的全局坐标,并与"手动分别采样再混合"的参考结果交叉验证(偏差 < 1e-5);
4. 全部骨骼插值局部/全局矩阵与蒙皮矩阵**行列式非零**(非奇异),0~6s 连续采样无 NaN。

## 目录结构

```
src/
  math.ts       手写 mat4 / 四元数(乘法、求逆、行列式、TRS 合成、slerp)
  skeleton.ts   骨骼 JSON 解析校验 + FK + 逆绑定矩阵
  animation.ts  片段采样、TRS 混合、idle/walk 状态机(Crossfade)
  renderer.ts   WebGL2 蒙皮渲染器、uniform 上限检测、贪心分批
  demoData.ts   演示骨架与 idle/walk 片段生成器(含旋转+缩放关键帧)
  selftest.ts   最小自测(Node 与浏览器共用)
  main.ts       浏览器入口:装配 + 渲染循环 + 页面自测输出
```
