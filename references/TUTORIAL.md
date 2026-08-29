# 讲解视频产线与难点插剪（有口播教学的视频才读本文）

主产线（SKILL.md §2）覆盖分析成片；本文是它的两个条件分支：
**A. 讲解片**（教练口播教学为主、攀爬只是演示）与 **B. 难点插剪**
（把说明片段剪进主线成片强调难点）。视觉规范两产线共用，见 SKILL.md §6。

## A. 讲解视频产线

管线：转录 → 字幕 → 姿态(演示段) → 技术动作 → 组合 → 渲染

```bash
PY=<conda env python>
# S1 转录（whisperX large-v3 GPU；首次需 nltk punkt 与 HF 镜像，见 TROUBLESHOOTING.md）
$PY scripts/transcribe.py <video> -o work/words.json --dict DICT.md
# S2 字幕：词边界断句 + 标点保留 + DICT 纠错映射 + 高亮（产物 SRT+JSON）
$PY scripts/caption_fix.py work/words.json -o work/captions.srt --json work/captions.json --dict DICT.md
# S3 姿态与分析同主产线（全片提取），技术检测只跑演示段
$PY scripts/tech_moves.py work/analysis.json work/tech.json --from-t <演示开始秒>
# S4 组合（--pose-start 前只有字幕，之后姿态层+徽章淡入）
$PY scripts/gen_tutorial.py work/analysis.json work/captions.json --video <video> \
  --out-dir <proj> --pose-start <秒> --tech work/tech.json \
  --title-main "..." --title-sub "..."
```

### 讲解片独有规则（视觉规范之外的增量）

> 注：分析成片产线已于 2026-08-28 移除跟随技术徽章（遮挡手脚/滞后/
> 与面板重复，见 SKILL.md §5）；讲解片模板暂保留徽章层，下次实际使用
> 讲解产线时按用户意向决定是否同样移除。

- **讲解段（--pose-start 之前）只有字幕**：无骨架/HUD/徽章/转移提示
- **转移徽章时间必须用绝对组合时间**（fmt_events 返回段内时间，要 +pose_t0；
  踩过：忘了偏移导致徽章出现在片头）
- 结尾小结卡显示技术动作次数/种类/转移次数（比 3pt 秒数更契合讲解；
  数字带单位）
- 口播与动作分析冲突时**以动作分析为准**

### DICT.md（工作区根目录，两层共用）

分节：`## 线路/人名`、`## 口播高频术语`、`## 纠错映射`（错词 => 正词）、
`## 转录 initial-prompt`。纠错映射随视频沉淀：通用词进 DICT.md 跨视频复用，
视频专属词（如把某词纠成特定线路号）只进该视频 work/fixes_*.json。

## B. 难点插剪

先预剪出编辑母版，再对母版整片重跑管线（姿态跟随能扛住剪辑切换，
双 track 会自动交接）：

```bash
# 1) 定位：主片难点口播时间（caption SRT 里"难点/难扣"句）+ 说明视频对应演示段
# 2) ffmpeg concat 母版（旋转烘焙 + 密集关键帧 + 音轨连续）：
ffmpeg -i 主片 -i 说明 -filter_complex "[0:v]trim=0:T,setpts=PTS-STARTPTS[v0];[0:a]atrim=0:T,asetpts=PTS-STARTPTS[a0];[1:v]trim=S:E,setpts=PTS-STARTPTS[v1];[1:a]atrim=S:E,asetpts=PTS-STARTPTS[a1];[0:v]trim=start=T,setpts=PTS-STARTPTS[v2];[0:a]atrim=start=T,asetpts=PTS-STARTPTS[a2];[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]" -map "[v]" -map "[a]" -c:v libx264 -preset fast -crf 18 -g 30 -keyint_min 30 -sc_threshold 0 -movflags +faststart -c:a aac -b:a 128k master.mp4
# 3) 对 master 重跑：transcribe（字幕按新时间轴重新转录，勿拼接旧字幕）
#    → extract/biomech/tech → gen_overlay **--seg 0:N-1 手动全选**
#    （插入段若把攀爬游程切开，pick_segment 会错截；手动 --seg 保全程）
```

- 插入点选主片口播的自然停顿（前句结束与「难点」句起始之间）；
- 说明段取完整演示（起手句 →「就大概就是这样」类收束句）；
- 说明视频可能被中途重命名，编码前先确认文件名存在。
