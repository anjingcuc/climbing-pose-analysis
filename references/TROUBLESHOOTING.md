# 排错手册：环境坑 + 门禁失败处置

按症状查表。**门禁（validate/hyperframes check）失败即中断管线，先修数据再继续。**

## 目录

1. [质量门禁失败](#质量门禁失败)
2. [环境与依赖坑](#环境与依赖坑)
3. [渲染/管线坑](#渲染管线坑)

## 质量门禁失败

| 失败信息 | 含义 | 处置 |
|---|---|---|
| `detection rate X < 75%` | 人物太小/太暗/严重遮挡 | 提高 `--imgsz 1536`、降 `--conf 0.15`；或裁切放大源视频 |
| 帧号 gap | 抽帧丢帧 | 用 ffmpeg 先转恒定 30fps 再提取 |
| `state X != expected` | 分析逻辑与接触数矛盾 | 代码 bug，跑 `pytest tests/` 定位，勿改数据迁就 |
| `com detected < 85%` | 骨架大面积丢失 | 同 detection rate；或检查选人是否锁错（看 debug_render） |
| `state_s sum != duration` | stats 汇总错误 | 代码 bug，跑测试 |
| check `content_overlap` | 标签/面板碰撞 | 模板内碰撞盒参数调整后重新生成 |
| check `gsap_exit_missing_hard_kill` | 退出动画跨越后续 clip 边界 | gen_overlay 已自动补边界 kill；复发则查 clip_starts 集合 |
| check 对比度 <4.5:1 | 小字过暗 | 提亮对应 CSS 色（如 #77828d→#97a2ad 级别） |

## 环境与依赖坑

- **torch 下载慢**（阿里云单线程）：并发分段下载（HEAD 取大小 → N 路 Range
  分块 + 断点续传 → 拼接，16 线程约 10MB/s），再 `pip install 本地.whl`；
  小依赖走清华 PyPI
- **whisperX 把 torch 升级为 CPU 版**：装完必须重装
  `torch==2.6.0+cu124 torchvision torchaudio`（阿里云镜像）并验证
  `cuda.is_available()`；pip 显式钉 `"torch==2.6.0+cu124"` 可防解析器乱动
- **ctranslate2 与 torch cuDNN 冲突**（转录报 `cudnn_ops_infer64_8.dll` 缺失）：
  pip 的 ctranslate2 4.4 要 cuDNN 8，torch 2.6 自带 cuDNN 9 → 升级
  `ctranslate2>=4.5`（whisperx 的 <4.5 上限可无视），并把
  `<env>/Lib/site-packages/torch/lib` 加进 PATH 再跑
- **torch 2.6 加载 wav2vec2 对齐权重报 omegaconf WeightsUnpickler 错**：
  加环境变量 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
- **ctranslate2 要 pkg_resources**：setuptools≥81 已移除，装 `setuptools<81`
- **whisperX 需要 nltk punkt**：预下载到 `~/nltk_data/tokenizers/`
  （gh-proxy 取 punkt.zip 与 punkt_tab.zip）
- **HF 模型下载 401（cas-server.xethub）**：新版 huggingface_hub 走 xet
  协议镜像不认，加 `HF_HUB_DISABLE_XET=1`；模型一律 `HF_ENDPOINT=https://hf-mirror.com`
- **yolo 权重下载**：GitHub 直连不通用 `https://gh-proxy.com/` 前缀；
  装完 `torch.load` 验证 kpt_shape=[17,3]（yolo11x-pose.pt = 118,481,010 字节）
- **环境残留代理拒绝连接**（如 127.0.0.1 端口不通）：pip 加 `NO_PROXY='*'`

## 渲染/管线坑

- **手机竖屏旋转元数据**（最重要的坑）：编码横屏 + `rotation=-90` → 显示竖屏。
  OpenCV ≥4.5 读帧自动应用旋转（关键点天然在竖屏坐标系）；ffmpeg 重编码把
  旋转烧进画面；hyperframes 画布必须等于**显示尺寸**（gen_overlay 用 cv2 首帧
  探测）。判定：`ffprobe -show_entries side_data=rotation` + cv2 首帧 shape。
  症状：横屏画布 + cover = 视频放大 1.78 倍 + 骨架完全错位
- **人物占比小时接触漏锁**：噪声底（手脚静止期 p25 速度，torso/s）可能 >
  默认 on_t → 2pt/1pt 虚高、事件全变 4pt→4pt。处置见 SKILL.md §参数速查
  的 on_t 校准流程
- **切片关键帧必须密集**（`-g 30 -keyint_min 30 -movflags +faststart`，
  gen_overlay 已内置）：hyperframes 逐帧 seek DOM video，x264 默认 250 帧
  GOP 在 4K 下 capture 阶段永久卡死（framesCompleted=0 十几分钟）
- **音轨**：`<audio>` 元素要求切片带音轨（`-c:a aac`，不可 `-an`）
- **多人帧**：攀爬者下方常有保护者 → `select_target` 用「最高优先 +
  髋部 250px 连续门」锁定攀爬者（有单测）
- **速度估计**：逐帧差分放大抖动，必须 ±5 帧中心位移 + 中值滤波
- **转移事件**：重心位移发生在「破接触→再接触」周期内，不在三点保持相位内找
- **SVG 隐藏文本**：被碰撞跳过的标签必须清空 textContent（全堆在 -99,-99
  会被布局审计判重叠）
- **Studio 挂死**：render 后 preview 守护进程常无响应：
  `netstat -ano | grep 3002` 找 PID → taskkill → 重启 preview；
  浏览器打不开用 `http://127.0.0.1:3002`
