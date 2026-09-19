<div align="center">

# gathervis

[English](https://github.com/zzzzswh/gathervis/blob/main/README.md) | **简体中文**

**面向 Python / GPU 服务器的地震道集 Web 查看器。**

*指向远程服务器上的 100 GB 数据文件，一秒钟在你的浏览器里打开。*

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-58%20passing-brightgreen)

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_gather.png" width="820" alt="Shot gathers 页：分析窗、窗内谱（十字线开启）与四个工具块"/>
<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_slice.png" width="820" alt="Volume slices 页：AGC 增益后的数据体，沿 shot 轴拉伸 8 倍的三维长方体"/>

*二维道集 QC 与三维体切片，同一个浏览器页面。*

</div>

> **💬 欢迎提需求！**
> 如果你希望它能做什么 —— 显示方式、处理步骤、文件格式 ——
> **请务必开 issue 告诉我，我会实现的，非常感谢！**

---

## 为什么做 gathervis？

* **超大数据秒开。** 文件以懒加载 memmap 方式打开——浏览一炮只读那一炮的字节，
  切一刀只读那一片。100 GB 的原始二进制和 1 MB 的文件打开一样快。
* **只需要一个浏览器。** 单端口服务。VS Code Remote-SSH 自动转发端口；Jupyter
  里直接内嵌渲染；裸 `ssh -L` 也行。
* **观测系统联动。** 给出炮点/检波点坐标就有 Geometry 页——点任意炮点直接跳到
  它的炮集。
* **cigvis 风格的三维视图。** 数据体可渲染为可旋转的长方体 + 三个实时切片平面
  （WebGL）。
* **内置 QC 工具。** wiggle / 变密度显示、行业色标、零相位 Ormsby 滤波、
  AGC / 道均衡、画窗谱分析（振幅谱 + f-k）、手动 / STA-LTA 自动初至拾取。
* **简单好用。** ~1000 行包代码、两个渲染原语、一种传输格式
  （抽稀 → uint8）。

## 安装

```bash
pip install -e .            # numpy + panel + bokeh + plotly
pip install deepwave        # 可选，跑正演示例用（会拉 torch）
pip install segyio          # 可选，SEG-Y 导入
```

## 快速上手

```python
import gathervis as gv

# —— 快速查看，无观测系统 ——
gv.show(gather_2d, dt=0.002, port=8080)          # 一个 (道, 时间) 面板
gv.show(line_3d, dt=0.002, port=8080)            # 默认 (shot, rec, time)：逐炮浏览
gv.show(line_3d, view='slices', port=8080)       # 同一数组作为三维长方体切片
gv.show('shots.bin', shape=(60,192,1200),        # 原始二进制：懒 memmap，秒开
        dtype='float32', dt=0.002, port=8080)
gv.show(vel, axes=('x','y','depth'), dt=10.0)    # 属性体（如速度模型 QC）

# —— 带观测系统 ——
ds = gv.from_array(data, src=src_xyz, rec=rec_xyz, dt=0.002)
gv.show(ds, port=8080)   # 多一个 Geometry 页；点炮点 -> 跳到该炮道集
```

命令行零代码：

```bash
gathervis line2d_data.npy --geom line2d_geom.npz
gathervis shots.bin --shape 60 192 1200 --dt 0.002
gathervis field.sgy                                # SEG-Y：dt 与观测系统读自道头
gathervis vel.npy --axes x y depth --dt 10 --cmap rainbow
```

这里有几个用合成数据的示例：`python examples/quickstart.py` 直接服务一个解析合成数据；
`python examples/deepwave_line2d.py` 用 [deepwave](https://github.com/ar4/deepwave)
正演一条二维测线；`python examples/deepwave_lines3d.py` 在一个丰富的三维速度模型
（倾斜层 + 背斜穹隆 + 低速河道透镜体）上用 deepwave 的三维引擎**打五条二维测线**，
跑完直接打开查看。

## 功能一览

### 逐炮浏览，与观测系统联动

**Shot gathers** 页（页首第一张图）拖动滑条实时翻炮，或在 *go to shot* 输入
炮号直接跳转。直接在道集上画分析窗、算窗内谱（十字线 + 实时频率/dB 读数方便
比峰）、手动拾取或 STA/LTA 自动拾取初至——面板下方的四个工具块
（*Window · Spectrum · Event picking · FB picking*）各管各的。

**Geometry** 页显示采集布设；点任意炮点即选中该炮并跳回其道集，激活炮的排列同步高亮：

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_geometry.png" width="820" alt="Geometry 页：五条二维测线，金色星标为激活炮，其排列以蓝色高亮"/>

### 操作速查

| 想做什么 | 怎么做 |
| --- | --- |
| 画矩形窗 | 工具栏选中框选工具 → **SHIFT+拖拽**（或点一下、移动、再点一下） |
| 画多边形窗 | 工具栏选中多边形工具 → 逐点点击加顶点，**双击或 ESC** 收尾 |
| 移动 / 删除窗 | 直接拖动；点选后按 **BACKSPACE**（或点 *clear windows*） |
| 加 / 移 / 删拾取点 | 工具栏点拾取工具 → 点击添加、拖动移动、点选 + **BACKSPACE** 删除（或 *clear shot* / *clear all*） |
| 自动拾初至 | 设好 STA / LTA / threshold → *auto pick*；再用 *snap*（peak / trough / \|max\|）或手动拖动精修 |
| 比较谱峰高度 | 工具栏十字线图标开准星，配合面板下方的 freq / dB 读数 |
| 单轴伸缩 | 工具栏的 x-only / y-only 滚轮缩放工具 |
| 调整三维长方体比例 | 每个维度各一条 stretch 滑条（1.0 居中，0.125×–8×） |
| 跳到某一炮 | 拖炮号滑条、*go to shot* 输入，或在 Geometry 页点炮点 |
| 调谱面板大小 | Spectrum 块里的 *size*（S / M / L） |
| 存图 | **⤓ full image** 按钮（原分辨率全图）· 工具栏 save 图标（当前视野，屏幕分辨率） |

同一份速查在应用里点 *? gestures* 按钮随时可看。

### 数据体的三维长方体切片（cigvis 风格）

任何三维数组——整条测线、一个三维炮集、一个速度模型——都渲染为线框长方体 +
三个轴对齐切片平面。拖动滑条移动切片（每步只传输那一片抽稀数据），自由旋转/缩放
（相机在任何更新后保持不动），用逐轴拉伸滑条（0.125×–8×，几何级数档位，1.0 居中）
重塑长方体——页首第二张图就是沿 shot 轴拉伸 8 倍、经同一条滤波/增益链做过
AGC 的测线。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/velocity_qc.png" width="440" alt="depth 轴速度模型 QC"/>

上图是同一套正演背后的三维速度模型，以 `axes=('x','y','depth')` 查看——
属性体自动使用深度标注与最小/最大值式色标范围。

### wiggle 或变密度，配上你真正在用的色标

一键在变密度与变面积 wiggle 之间切换所有二维面板，*flip polarity* 复选框整体翻转
极性（wiggle 充填瓣与变密度红蓝同步互换——SEG normal ↔ reverse）。色标：
`seismic`、`gray`、`petrel`（锚点取自 cigvis）、`rainbow`，各配 `_r` 反转变体。
clip percentile 滑条设定色标范围——同时兼作 wiggle 增益。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/display_modes.png" width="620" alt="变密度 vs wiggle"/>

### 零相位梯形滤波与增益

道集视图上的经典带斜坡（Ormsby 式）**低通 / 高通 / 带通**：`f1–f2`（低切）与
`f3–f4`（高切）线性斜坡，严格零相位；外加 **AGC**（滑动窗 RMS，窗长可调）与
**道均衡**——处理链为滤波 → 增益，增益开启时色标范围自动重算。逐炮按需处理，
memmap 懒加载不受影响。Volume-slices 页跟随同一条链：512 MB 以内的数据体整体
处理一次（链条变化时），三个切片严格一致；更大的 memmap 回退为原始数据并显示
提示。若传入 CuPy 数组，全部处理透明地在 GPU 上运行。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/filters.png" width="620" alt="带通滤波压制含噪道集"/>

### 画窗 → 谱 → 导出

直接在道集上画**矩形**（SHIFT+拖拽，或点-移-点）或**多边形**（逐点点击，ESC
收尾）分析窗，点 *compute spectrum*：对门控后的样点做纯 DFT，画 `|rfft|` 相对
本窗峰值的 dB。不加窗函数、不做归一、不做任何平滑——窗函数本身就是对频谱做
卷积（周期 Hann 严格等于对复谱做 `(-1/4, 1/2, -1/4)` 三点卷积），所以不用。
`traces` 选窗内各道怎么进图：**per trace**（逐道画，不做任何合并）、**mean**、
**middle trace**；`y axis` 切换线性**振幅**（归一到最响的那条曲线峰值为 1.0，
主频一眼可见）和 **dB**（把弱尾巴和噪声底拉开看）。图例写明画的是什么，以及
频率间隔 `df = 1/时窗长度`。

「不做处理」有两个必然的代价，它们是矩形截断后 DFT 的固有性质，不是 bug：
矩形门的旁瓣只按 1/f 衰减，所以大约低于峰值 40 dB 之后，你读到的是窗而不是
数据；单道周期图只有 2 个自由度，它的抖动是真实的，且不随记录变长而减小。
多道平均把抖动压掉 `sqrt(N)` 倍——但仅对**相互独立**的道成立；在相干性好的
道集上各道近乎复制品，平均值和单道几乎没区别。窗定义可**导出 / 导入
JSON**。信号窗 vs 噪声窗的对比十秒钟搞定。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/windows_spectra.png" width="720" alt="分析窗及其谱"/>

### 下载原图

面板上屏之前会按像素预算做 stride 抽稀，而 bokeh 自带的 save 工具存的是 canvas
截图、尺寸取决于图当前在屏幕上多大——两者都给不了"按数据自身分辨率"的图。
*fit* / *fullscreen* 右边的 **⤓ full image** 按钮可以：它把道集按一道一像素、一采样一像素
重新渲染，下载无损 PNG，色标、变密度 / wiggle、剪切百分位、极性、梯形滤波、
AGC / 道均衡全部与屏幕一致。wiggle 模式下画的是**全部道**，而不是浏览器拿到的
那个子集。不依赖 matplotlib、不用无头浏览器——PNG 由 numpy + 标准库 zlib 写出。

想要局部就用工具栏自带的 save 图标（当前视野，屏幕分辨率）。

脚本里批量出图同理：

```python
from gathervis.process import bandpass, agc
from gathervis.render import save_png

a = agc(bandpass(g.shot(7), g.dt, f3=60, f4=80), g.dt)
save_png("shot007.png", a, cmap="gray")            # 或 display="wiggle"
```

### 同相轴拾取

工具栏点拾取工具——点击添加、拖动微调、BACKSPACE 删除——虚线按道号连接拾取点，
走时形态一目了然。**auto pick** 运行 gapped STA/LTA 初至拾取（STA / LTA / 阈值
可调，带抗尖峰的持续触发判据），作用于当前显示（即滤波后）的道集；**snap** 把
任意拾取吸附到该道最近的波峰 / 波谷 / |最大值|。拾取按炮存储、随浏览跟随；
**导出 / 导入 CSV**（`shot,trace,time_s`），可直接喂给层析 / FWI 流程。

### SEG-Y 导入

`gv.from_segy('field.sgy')` 或直接 `gathervis field.sgy`——dt 读自二进制卷头，
炮按 FFID 分组，炮检点坐标（含 SEG-Y 标量规则）自动进入 Geometry 页；非规则炮
补零并警告。需要可选依赖 `segyio`。

## 远程使用（三选一）

1. **VS Code Remote-SSH（推荐）。** 在集成终端里跑任意示例；VS Code 自动转发
   端口，打印出的 `http://localhost:8080` 直接可点。零配置。
2. **Jupyter。** 省略 `port=`，返回的 app 直接在 notebook 里内嵌渲染——
   完全不需要额外端口。
3. **裸终端。** `ssh -L 8080:localhost:8080 user@gpu-server`，然后本地打开
   URL。端口被占用时 gathervis 自动换一个空闲端口并打印出来。

## 语义（唯一要记住的规则）

最后一维永远是纵轴：数据用 **time**，属性体用 **depth**。裸三维数组默认为
`(shot, rec, time)` 并逐炮浏览；传 `view='slices'` 改为体切片查看，或用 `axes=`
显式声明其他语义。`axes` 说明数组*是什么*，`view` 说明*怎么看*——二者独立。

## 设计笔记

* **处处懒加载。** memmap 数据；浏览一炮只读那一炮的字节；切片只读那一片；
  滤波与谱分析只作用于当前显示的道集。
* **廉价传输格式。** 面板在服务端按像素预算跨步抽稀并量化为 uint8 后才上线。
* **两个渲染原语。** 二维面板是 bokeh 图像 / wiggle；数据体是 plotly WebGL
  切片平面。二者共用同一条抽稀 → uint8 管线。
* **实时拖动。** 滑条拖动中实时更新；三维相机、拉伸倍率、已画的窗在任何更新后
  全部保留。每个二维面板都带 x / y 单轴滚轮缩放和光标读数（道号、时间、鼠标
  所在像素的振幅，纯客户端运行），谱面板带同款读数与可开关的十字线。

## 路线图

**M2**（剩余）：联动对比面板 · 客户端体缓存（cigvis 级切片拖动手感）
（AGC / 道均衡 / CuPy：已完成）。**M3**（剩余）：道头索引、按任意键抽道集
（SEG-Y 导入：已完成）。**M4**（剩余）：共偏移距/时间切片、NMO 预览
（同相轴拾取：已完成）。完整规划见 [`docs/plan.md`](https://github.com/zzzzswh/gathervis/blob/main/docs/plan.md)。
*以及任何你提出的需求——见页首。*

## 致谢

三维切片视图沿用 [cigvis](https://github.com/JintaoLee-Roger/cigvis) 的交互风格
（并借用其 `petrel` 色标锚点，MIT）。正演示例使用
[deepwave](https://github.com/ar4/deepwave)。基于
[Panel](https://panel.holoviz.org/)、[Bokeh](https://bokeh.org/) 与
[Plotly](https://plotly.com/javascript/) 构建。

## 许可证

MIT
