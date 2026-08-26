---
name: climbing-pose-analysis
description: 对固定机位攀岩/抱石视频做本地 GPU 姿态追踪与生物力学分析，渲染竖屏叠加标注成片（骨架/重心轨迹/接触检测/三点平衡状态机/支撑三角/重心转移事件/技术动作徽章/语音字幕）。触发：用户提到「攀岩视频 姿态分析」「三点平衡 重心转移」「抱石 技术分析 骨架标注」「攀岩 CoM 分析成片」，或要求分析攀爬动作、给攀岩视频叠加技术标注与字幕。不适用：移动机位跟拍；需要 3D 关节力矩/肌肉力的科研场景（→ OpenSim/Pose2Sim）。
---

# Climbing Pose Analysis · 攀岩姿态与三点平衡分析管线

固定机位攀岩视频 → **GPU 姿态追踪** → **生物力学分析**（重心/接触/三点平衡）→
**hyperframes 叠加成片**。每个阶段有质量门禁；脚本与测试在 `scripts/`、`tests/`
自包含；出错查 [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md)。

## 0. 前置检查（缺一先补）

| 依赖 | 检查 | 备注 |
|---|---|---|
| GPU | `nvidia-smi` | 任意 CUDA 显卡；3090 上 5min 视频提取约 3 分钟 |
| conda env | `python -c "import torch;print(torch.cuda.is_available())"` | Python 3.11 + `torch==2.6.0+cu124` + `ultralytics` + `scipy` + `opencv-python` + `pytest` |
| ffmpeg | `ffmpeg -version` | 切片/抽帧/转码 |
| Node ≥22 | `node --version` | hyperframes CLI |

pip 镜像（中国网络）：torch 走 `-f https://mirrors.aliyun.com/pytorch-wheels/cu124/`
（单线程慢 → 并发分段下载，见 TROUBLESHOOTING.md），其余走清华 PyPI。
yolo11x-pose 权重（118,481,010 字节）GitHub 直连不通时加 `https://gh-proxy.com/`
前缀，装完 `torch.load` 验证 kpt_shape=[17,3]。

## 1. 快速开始

```bash
PY=<conda env python>
$PY scripts/run_pipeline.py <video.mp4> \
  --model yolo11x-pose.pt --title "线路名 · 三点平衡 / 重心转移分析" \
  --work-dir <proj>/work --out-dir <proj>/overlay
cd <proj>/overlay && npx hyperframes@0.8.6 check && npx hyperframes@0.8.6 render
```

管线内部：提取 → `validate pose` 门禁 → 分析 → `validate analysis` 门禁 →
选段+生成 overlay 项目。**完成判据**：`renders/*.mp4` 已产出，且满足——
check 0 error；snapshot 抽帧骨架贴合攀岩者；首帧为原始画面+局部标题卡。

## 2. 分阶段流程与质量门禁

| 阶段 | 命令 | 产出 | 门禁（不过则停） |
|---|---|---|---|
| S1 姿态提取 | `scripts/extract_pose.py <video> -o pose.json --model <pt> [--imgsz 1536]` | 逐帧 COCO-17 关键点 + 单目标跟随 | `scripts/validate.py pose pose.json --min-detect 0.75 --w W --h H`：检出率≥75%、帧号连续、坐标在画面内 |
| S2 生物力学 | `scripts/biomech.py pose.json analysis.json [--on-t X]` | 逐帧角度/CoM/接触/状态 + 转移事件 + stats | `scripts/validate.py analysis analysis.json --min-com 0.85`：schema 全键、状态与接触数一致、CoM 检出≥85% |
| S3 选段+叠加层 | `scripts/gen_overlay.py analysis.json --video <video> [--tech tech.json] [--captions captions.json] [--seg a:b] [--title ...]` | `overlay/{index.html,data.js,segment.mp4}`（画布=源显示分辨率） | hyperframes check 0 error |
| S4 渲染 | `check` → `snapshot --at ...` 目检 → `render` | 源分辨率竖屏 MP4 | snapshot 确认骨架在身上；交付用 `--quality high` |

调试：`scripts/debug_render.py analysis.json <video> out.mp4`（OpenCV 快速版）。
有口播的视频在 S3 前接字幕产线（§4）。

## 3. ⚠️ 每跑必知：手机竖屏视频的旋转元数据

手机竖屏视频 = 横屏编码 + `rotation=-90` 元数据 → **显示为竖屏**：

- OpenCV ≥4.5 读帧自动应用旋转，关键点天然在竖屏显示坐标系；
  ffmpeg 重编码切片会把旋转烧进画面；切片**不缩放**
- hyperframes 画布**必须等于视频显示尺寸**（gen_overlay 用 cv2 首帧自动探测，
  如 2160×3840）；HUD 按 1080×1920 基准网格布局，`scale = min(W/1080, H/1920)`
  等比放大（横屏取两轴最小值，防面板吃掉半屏）
- 判定：`ffprobe -show_entries side_data=rotation` + cv2 首帧 shape
- 症状：横屏画布 + cover = 视频放大 1.78 倍 + 骨架完全错位

其余坑（接触漏锁/多人帧/4K 渲染卡死/whisperX 环境等）见
[TROUBLESHOOTING.md](references/TROUBLESHOOTING.md)，按症状查表。

## 4. 语音字幕与技术动作（有口播的视频）

```bash
$PY scripts/transcribe.py <video> -o work/words.json --dict DICT.md   # L1
$PY scripts/caption_fix.py work/words.json -o work/captions.srt --json work/captions.json --dict DICT.md   # L2
$PY scripts/caption_llm.py work/captions.json DICT.md -o work/captions_llm.json   # L3 可选，无 key 自动跳过
$PY scripts/tech_moves.py work/analysis.json work/tech.json [--from-t 秒]
```

- **DICT.md**（工作区根目录）：线路/人名、口播术语、「纠错映射」（错词 => 正词，
  随视频沉淀、跨视频复用；视频专属词只进该视频 work/fixes_*.json）、
  「转录 initial-prompt」
- **三层纠错**：L1 transcribe --dict 喂解码器词汇提示；L2 caption_fix --dict
  应用同音词典+纠错映射；L3 caption_llm（GLM，key 从 `--api-key`/
  `$ZHIPUAI_API_KEY`/工作区或技能根 `.env` 的 `ZHIPUAI_API_KEY=` 自动读取，
  占位符忽略）做剩余错别字，difflib≥0.8 改写守卫、时间轴永不动
- **字幕断句双门禁**：词内绝不拆分（只按词元切+语气词回溯，单测守护）；
  停顿 ≥0.30s 加逗号、句末（≥0.55s 或段尾）加句号
- 技术动作检测器（14 个相位/事件 + 不检测清单 + 调参纪律）见
  [TECHNIQUES.md](references/TECHNIQUES.md)
- **讲解片成片 / 难点插剪**（说明片段剪进主线强调难点）见
  [TUTORIAL.md](references/TUTORIAL.md)

## 5. 视觉规范（用户已确认的默认，两产线统一遵守，勿退回）

**开场与结构**：
- **选段起点=第 0 帧**：准备过程完整保留，只裁攀爬结束后的尾部；分析图层
  从攀爬开始才出现
- **第一帧 = 原始画面 + 局部标题面板**：禁止全屏蒙版、禁止入场动画；
  标题面板半透明毛玻璃局部卡片、opacity 1 起始、~4.2s 淡出
- 分析图层以 `pose_t0 = 首个 climbing 帧`为门：提前 0.8s 挂载、
  从 pose_t0 起 1.2s 淡入；之前画面完全干净
- **底部 HUD 面板 = 支撑点 + 技术动作**：左半四瓷砖（支撑=绿实框、移动肢体=
  黄虚框「移动」，0.25s 消抖）；右半当前技术动作（持续相位+实时计时 /
  最近闪现「上一动作 · Ns 前」/ 兜底「常规攀爬」）。顶部只留状态 pill + 时钟，
  **pill 状态从消抖后的瓷砖推导**（三个模块永不同屏打架）
- 底部面板遮挡人物时自动隐藏（关键点判入场+横向范围限制，滞回 0.4s/0.6s；
  事件字幕同步隐藏）
- 结尾小结卡 4 项**带单位**：3pt 时长（秒）/ 转移（次）/ 技术动作（个）/
  最大侧倾力臂（身位）；标题图例注明「身位=躯干长」

**视觉层级（三点平衡主线是主角）**：
- 骨架 0.22 透明度纯背景；**不显示关节角度标注**
- 支撑三角为主角：描边 6、fill 0.20、柔光回声层；3pt 接触点亮绿脉动圈；
  移动肢体=虚线暗白圈；其余半透明红圈 0.45；重心点 r13 + r32 光晕
- **技术徽章**跟随头顶上方 1.15 身位（0.5s 平滑，y∈[370,1200]×SCALE 钳制）；
  **锚点与任一关节/手脚点碰撞时按 上→左→右→下 错位搜索避让**；
  持续徽章 31px、闪现徽章 38px
- **字幕位置稳定 v3**：`cap_zone_windows` 骨骼段碰撞状态机——字幕矩形与
  12 段骨骼线段+关键点精确相交（bbox 相交不算），当前槽位真实碰撞持续
  1.0s 才切换（顶槽 200↔低槽 1280）；绝不移入对方也占用的槽位；每槽位
  最短驻留 6s；常态顶部区、标题面板期间 470 下；严格一次一条、
  小结卡前截断、术语绿色高亮；勿固定放底部
- 事件字幕：3pt→3pt 链标注「三点平衡转移」，其余「重心转移」；排队不丢弃

模板在 `scripts/overlay_template.html`，**改模板后重新生成，勿手改生成的
index.html**。（历史教训：规则只写在一条产线里导致另一产线整批丢失——
故本节为两产线共用。）

## 6. 静默执行（防黑窗，Windows 必须）

- Python 调子进程一律 `procutil.run()`（自动 CREATE_NO_WINDOW）
- hyperframes/npx 整树无弹窗：`python scripts/hf.py --cwd <proj> -- check|render`
  （PowerShell 隐藏控制台继承给整棵进程树；windowsHide 管不到孙进程）

## 7. 参数速查

- `--seg a:b`：手动片段帧区间；默认起点=第 0 帧，终点=最长攀爬段结束 +1.5s
- `--max-seg-s`（默认 0=不封顶）：总时长上限（从尾部截短）；4K 渲染约 1.6min/75s
- `--tech tech.json` / `--captions captions.json`（gen_overlay）：徽章层 / 语音字幕层
- **接触阈值 on_t 校准（人物小/噪声大时必做）**：测静止期手脚 p25 速度
  （`detect_contacts` 返回的 speed），`--on-t ≈ 1.4×噪声底`；用 debug_render/
  抽帧目检接触圈贴合真实踩点，再对照口播时间戳复核
- 事件阈值：位移 >0.20 身位、破接触前 ≥6/8 帧接触、单段最多 60 条
- 渲染质量：draft 迭代、high 交付；gsap 可离线：放 `workspace/gsap.min.js`
  （gen_overlay 自动拷贝，缺失回退 CDN）

## 8. 工程纪律与 LLM 参与最小化

原则：**能用确定性代码+测试守护的，绝不交给 LLM 临场发挥**——省 token
更重要是产出一致、可回归。LLM 只保留两处：可选的字幕 L3 纠错、
交付前一次的 vision 成片复核（固定 10 帧覆盖、固定提示词）。

**产出一致性规则**：
1. 改 `scripts/` 必先全绿 `pytest tests/ -q`（选人/清洗/角度/CoM/接触/选段/
   字幕断句/避让/验证器/端到端全覆盖），再真数据复核，再出片；
2. 人工纠错具通用性 → 当场沉淀进 DICT.md 映射；视频专属词只进该视频 fix 文件；
3. 视觉复核的可执行项当场修模板并回归测试；
4. 阈值改动在 [TECHNIQUES.md](references/TECHNIQUES.md) 留一行依据
  （为什么是这个数）；
5. 检测器调参遵守 TECHNIQUES.md 的调参纪律（口播时刻必须检出、不得泛滥）。
