# Climbing Pose Analysis · 攀岩姿态与三点平衡分析管线

[English](#english) | 中文

对固定机位攀岩/抱石视频做**本地 GPU 人体姿态追踪与生物力学分析**，并渲染竖屏
叠加标注成片。可作为命令行管线使用，也可作为 Agent 技能（SKILL.md）装载。

## 功能特性

- **姿态追踪**：YOLO11x-pose 全片提取 COCO-17 关键点，单目标跟随（多人帧自动锁定攀爬者）
- **生物力学分析**：Winter 人体测量学全身重心（CoM）+ 轨迹、腕/踝接触检测
  （速度滞回 + 噪声底自适应校准）、三点平衡状态机、支撑三角、重心转移事件
- **技术动作检测**（14 个，2D 几何可靠集，附阈值依据表）：侧身 / 挂脚 / 换脚 /
  交叉脚 / 交叉手 / 并手 / 旗式 / 直臂休息 / 高脚 / 内扣膝 / 锁臂 / 膝勾 /
  压重心 / 动态跳跃 / 翻上——实时显示在底部「当前技术动作」面板
- **语音字幕 v3（双引擎各司其职）**：FunASR paraformer-zh（中文 CER 优于
  whisper、自带 ct-punc 标点、热词硬偏置）出文本，whisperX large-v3 同音频
  出时间轴，difflib 文本锚点分段线性 warp 修正 FunASR 时间戳漂移；词边界
  断句 + 标点保留契约 + 停顿标点兜底 + 同音纠错词典（DICT.md）+ 可选
  LLM 纠错（五道确定性门禁）
- **克制的视觉标注**（不遮挡攀岩者）：低透明度骨架与支撑三角、三点平衡
  接触圈、重心光晕、底部支撑点四瓷砖面板；**无跟随徽章、无关节角度标注**
- **成片渲染**：hyperframes 渲染源分辨率竖屏 MP4；字幕采用骨骼段碰撞式
  稳定避让（真实遮挡才换位、最短驻留、绝不移入占用槽）；切片与编码自动
  探测 NVENC（驱动 ≥610 生效，回退 x264/CPU）
- **质量门禁**：pose / analysis 两级 schema 验证 + hyperframes check +
  **121 个单元测试**（含破坏性注入与真实语料回归契约）

## 环境要求

- NVIDIA GPU（CUDA）+ Python 3.11
- `torch==2.6.0+cu124` + `ultralytics` + `scipy` + `opencv-python` +
  `funasr` + `jieba` + `pytest`
- `ffmpeg`、Node.js ≥ 22（hyperframes CLI）
- yolo11x-pose.pt 权重（自行下载，勿入库）

## 快速开始

```bash
git clone https://github.com/anjingcuc/climbing-pose-analysis.git
cd climbing-pose-analysis

PY=<你的 conda env python>
$PY scripts/run_pipeline.py <video.mp4> \
  --model yolo11x-pose.pt --title "线路名 · 三点平衡 / 重心转移分析" \
  --work-dir work --out-dir overlay
cd overlay && npx hyperframes@0.8.6 check && npx hyperframes@0.8.6 render
```

有口播的视频接字幕产线（`transcribe --dict` → `caption_fix --dict` →
可选 `caption_llm` → `gen_overlay --captions`）；完整流程（分阶段门禁、
讲解片产线、难点插剪、排错手册）见 [SKILL.md](SKILL.md) 与
[references/](references/)。

## 作为 Agent 技能安装

把本仓库放进你的技能目录（如 `~/.agents/skills/`）即可被支持 SKILL.md
规范的 Agent（Claude Code / Codex CLI 等）按描述自动调用：

```bash
git clone https://github.com/anjingcuc/climbing-pose-analysis.git \
  ~/.agents/skills/climbing-pose-analysis
```

## 测试

```bash
python -m pytest tests/ -q    # 121 passed
```

覆盖：选人策略、关键点清洗、角度计算、Winter 质量模型、接触检测滞回、
三角形裕度、选段、字幕断句/标点保留/微碎句合并/跨界伪标点剥离、字幕避让
状态机、FunASR 词映射与漂移 warp、LLM 纠错五道门禁、验证器破坏性注入、
技术动作检测器、合成场景端到端。

## 项目结构

```
├── SKILL.md               # Agent 技能入口（管线流程 + 视觉规范 + 参数速查）
├── scripts/               # 管线脚本（提取/分析/字幕/动作检测/生成/渲染/对齐）
├── tests/                 # 121 个单元测试
└── references/
    ├── TECHNIQUES.md      # 技术动作检测器阈值与依据
    ├── TROUBLESHOOTING.md # 环境坑 + 门禁失败处置（按症状查表）
    └── TUTORIAL.md        # 讲解片产线 + 难点插剪
```

## 许可

[MIT](LICENSE) © anjingcuc

---

<a id="english"></a>

# Climbing Pose Analysis (English)

Local GPU pipeline for climbing/bouldering videos: pose tracking →
biomechanics (Winter CoM, contact detection, three-point-balance state
machine, weight-transfer events) → 14 climbing-technique detectors →
captioned, annotated vertical video at source resolution.

Caption engine v3: FunASR (paraformer-zh + ct-punc: better Chinese CER,
native punctuation, hotword bias) provides text while whisperX large-v3
times the same audio; difflib text anchors warp FunASR's drifting
timestamps onto the whisper timeline. Word-boundary segmentation with a
punctuation-retention contract, a homophone correction dictionary
(DICT.md) and an optional LLM pass behind five deterministic gates.

Visuals stay out of the climber's way: low-opacity skeleton and support
triangle, pulsing 3pt contact rings, a bottom support-points + current-
technique panel (no following badges, no joint-angle labels), and
skeleton-collision-driven subtitle placement that moves only on real
occlusion. Segment cuts and final encoding auto-probe NVENC.

Works as a CLI pipeline or as an Agent skill (SKILL.md). See
[SKILL.md](SKILL.md) for the full workflow and [references/](references/)
for detector thresholds, troubleshooting and the tutorial/crux-cut
pipelines. 121 unit tests. MIT licensed.
