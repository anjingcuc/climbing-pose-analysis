# Climbing Pose Analysis · 攀岩姿态与三点平衡分析管线

对固定机位攀岩/抱石视频做**本地 GPU 人体姿态追踪与生物力学分析**，渲染竖屏
叠加标注成片。可作为命令行管线使用，也可作为 Agent 技能（SKILL.md）装载。

```
视频 → 姿态提取(YOLO11x-pose) → 生物力学(Winter CoM/接触/三点平衡)
     → 技术动作检测(14 种) → 语音字幕(双引擎) → hyperframes 渲染(4K 竖屏)
```

## 功能特性

- **姿态追踪**：全片 COCO-17 关键点，单目标跟随（多人帧自动锁定攀爬者）
- **生物力学**：Winter 全身重心（CoM）+ 轨迹、腕/踝接触检测（速度滞回 +
  噪声底自适应校准）、三点平衡状态机、支撑三角、重心转移事件
- **技术动作检测**（14 个，2D 几何可靠集，阈值附依据表）：侧身 / 挂脚 /
  换脚 / 交叉脚 / 交叉手 / 并手 / 旗式 / 直臂休息 / 高脚 / 内扣膝 / 锁臂 /
  膝勾 / 压重心 / 动态跳跃 / 翻上——实时显示在底部「当前技术动作」面板
- **语音字幕（双引擎各司其职）**：FunASR paraformer-zh（中文识别更准、
  自带标点、热词硬偏置）出文本；whisperX large-v3 同音频出时间轴，文本
  锚点 warp 修正 FunASR 的时间戳漂移；词边界断句、标点保留契约、同音
  纠错词典（DICT.md）、可选 LLM 纠错（五道确定性门禁）
- **克制的视觉标注**（让位于攀岩者）：低透明度骨架与支撑三角、三点平衡
  接触圈、重心光晕、底部支撑点面板；无跟随徽章、无关节角度标注
- **质量门禁**：pose / analysis 两级 schema 验证 + hyperframes check +
  121 个单元测试（含破坏性注入与真实语料回归契约）

## 环境要求

| 依赖 | 说明 |
|---|---|
| NVIDIA GPU（CUDA）| 3090 上 5 分钟视频提取约 3 分钟；切片/编码自动探测 NVENC |
| Python 3.11 | `torch==2.6.0+cu124` + `ultralytics` + `scipy` + `opencv-python` + `funasr` + `jieba` + `pytest` |
| ffmpeg | 切片/抽帧/转码；NVENC 需驱动 ≥610 |
| Node.js ≥ 22 | hyperframes CLI（`npx hyperframes@0.8.6`） |

> [!NOTE]
> yolo11x-pose.pt 权重（118 MB）需自行下载，勿入库；GitHub 直连不通时在
> 下载地址前加 `https://gh-proxy.com/` 前缀。首次运行 FunASR 会从
> modelscope 下载约 1 GB 模型。

> [!IMPORTANT]
> 手机竖屏视频是「横屏编码 + rotation 元数据」——画布必须等于**显示尺寸**
> （管线用 cv2 首帧自动探测），否则骨架整体错位。判定与处置见
> [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md)。

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

> [!TIP]
> 中国网络环境：pip 走清华镜像，torch 走阿里云 pytorch-wheels（单线程慢时
> 用并发分段下载）；`HF_ENDPOINT=https://hf-mirror.com`。全部镜像与代理
> 处置见 TROUBLESHOOTING.md。

## 作为 Agent 技能安装

把本仓库放进技能目录（如 `~/.agents/skills/`）即可被支持 SKILL.md 规范的
Agent（Claude Code / Codex CLI 等）按描述自动调用：

```bash
git clone https://github.com/anjingcuc/climbing-pose-analysis.git \
  ~/.agents/skills/climbing-pose-analysis
```

## 测试

```bash
python -m pytest tests/ -q    # 121 passed
```

覆盖：选人策略、关键点清洗、Winter 质量模型、接触检测滞回、三角形裕度、
选段、字幕断句/标点保留/微碎句合并/跨界伪标点剥离、字幕避让状态机、
FunASR 词映射与漂移 warp、LLM 纠错五道门禁、验证器破坏性注入、技术动作
检测器、合成场景端到端。

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

---

English: a local-GPU pipeline for fixed-camera climbing videos — pose
tracking, biomechanics (Winter CoM, contact detection, three-point-balance
state machine), 14 technique detectors, and a dual-engine Chinese caption
pipeline (FunASR text + whisper timing, warped onto one timeline) —
rendered as annotated vertical video via hyperframes, with quality gates
and 121 unit tests. Works as a CLI or as an Agent skill (see SKILL.md).
