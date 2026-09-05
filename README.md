# Video Sheet Extract v2.0

从教学视频中自动提取滚动 TAB 谱/五线谱，去重合并后生成 A4 多页高清 PDF。

## 功能特性

- **智能乐谱检测**: 自动识别视频中的乐谱画面，过滤非乐谱帧
- **画面优化**: 去色块、背景平整、中位数合成，保留染色音符墨迹
- **三级去重**: 相邻折叠 → 全局合并 → 保险去重，高效消除重复帧
- **自适应排版**: 标题/副标题自动缩字号适配页宽，单行显示不换行
- **多种裁剪**: 支持自动检测、像素坐标、比例裁剪三种模式
- **乐谱类型**: 自动检测五线谱/六线谱(TAB)/五六线谱对照

## 环境要求

- Python 3.8+
- ffmpeg (系统需安装并加入 PATH)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行模式

```bash
python video-sheet-extract.py 视频文件.mp4 -o 输出.pdf [选项]
```

### GUI 模式

```bash
python video-sheet-gui.py
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `video` | - | 视频文件路径 (必填) |
| `-o, --output` | - | 输出 PDF 路径 (必填) |
| `-t, --title` | `""` | 标题 (单行自适应字号) |
| `-s, --subtitle` | `""` | 副标题 (单行自适应字号) |
| `--notation` | `auto` | 乐谱类型: `auto`/`staff`/`tab`/`both` |
| `--crop` | - | 比例裁剪 `起始:结束`，如 `0.55:1.0` |
| `--crop-y` | - | 像素裁剪 `y0:y1`，如 `120:600` |
| `--auto-crop` | - | 自动检测乐谱区域 |
| `--start` | - | 起始时间 `HH:MM:SS` 或秒数 |
| `--end` | - | 结束时间 `HH:MM:SS` 或秒数 |
| `--fps` | `0.5` | 采样频率 (次/秒) |
| `--dpi` | `300` | 输出 PDF 分辨率 |
| `--margins` | `15:15:10:10` | 页边距 mm `上:下:左:右` |
| `--percentile` | `0.5:99.5` | 对比度拉伸百分位 `低:高` |
| `--sharpen` | `1.2` | 锐化强度 |
| `--sharpen-sigma` | `1.0` | 锐化核大小 |
| `--group-t` | `0.95` | 第一级分组阈值 |
| `--merge-t` | `0.95` | 第二级合并阈值 |
| `--dedup-t` | `0.65` | 第三级去重阈值 |
| `--keep-temp` | - | 保留临时文件夹 |
| `--delete-temp` | - | 删除临时文件夹 |
| `--debug` | - | 打印调试信息 |

## 示例

### 基础用法

```bash
python video-sheet-extract.py input.mp4 -o output.pdf
```

### 指定标题和乐谱类型

```bash
python video-sheet-extract.py input.mp4 -o output.pdf -t "曲目名称" --notation tab
```

### 自定义裁剪和时间范围

```bash
python video-sheet-extract.py input.mp4 -o output.pdf --crop-y 200:800 --start 00:01:00 --end 00:05:00
```

### 高 DPI 输出

```bash
python video-sheet-extract.py input.mp4 -o output.pdf --dpi 600 --margins 12:12:10:10
```

## 项目结构

```
2.0/
├── video-sheet-extract.py   # 核心提取引擎
├── video-sheet-gui.py       # GUI 界面
├── requirements.txt         # Python 依赖
└── README.md               # 项目说明
```

## 技术说明

### 画面处理流程

1. **采样检测**: 按指定频率采样帧，检测是否包含乐谱内容
2. **画面优化**: 去色块、背景平整、中位数合成
3. **三级去重**: 消除重复帧，保留不同乐谱段
4. **增强放大**: 百分位对比度拉伸 + 锐化 + Lanczos 放大
5. **A4 排版**: 自适应页面布局，生成高清 PDF

### 乐谱检测算法

- 白底宏观统计: 白色像素占比、行覆盖率、墨迹占比联合判定
- 支持五线谱、六线谱(TAB)、五六线谱对照三种类型自动检测

### 去重算法

- **第一级 (相邻折叠)**: 相似连续帧合并为一组
- **第二级 (全局合并)**: 跨区域重复帧合并
- **第三级 (保险去重)**: 最终去重，确保无遗漏

## 临时文件说明

处理完成后会生成 `{输出文件名}_temp/` 文件夹:

```
_temp/
├── 01_raw_clean/   # 采样命中乐谱帧
├── 02_groups/      # 去重后每组代表帧
├── 03_strips/      # 增强后的乐谱条
└── 04_pages/       # 排版后的每页 PNG
```

使用 `--keep-temp` 保留，`--delete-temp` 删除。

## 许可证

MIT License
