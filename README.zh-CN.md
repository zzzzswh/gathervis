<div align="center">

# gathervis

[English](https://github.com/zzzzswh/gathervis/blob/main/README.md) | **简体中文**

**专业的地震叠前数据可视化 Web 查看器。**

*在浏览器里打开远程服务器上的叠前数据，像用处理软件一样翻炮、滤波、算谱、拾取。*

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-313%20passing-brightgreen)

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_gather.png" width="820" alt="Shot gathers 页：道集上方是翻炮与视图控件，侧栏按显示、处理、工具分组"/>
<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/wiggle.png" width="820" alt="Shot gathers 页：wiggle 显示方法，自定义道密度"/>
<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_slice.png" width="820" alt="Volume slices 页：三个切片平面，当前切片有描边，上方是移动切片的按键"/>

*二维道集 QC 与三维体切片，同一个浏览器页面。*

</div>

> **💬 欢迎提需求！**
> 如果你希望它能做什么 —— 显示方式、处理步骤、文件格式 ——
> **请务必开 issue 告诉我，我会实现的，非常感谢！**

---

## 为什么做 gathervis？

起因很实际：不想再用 matplotlib 一张张画叠前数据了。换一炮重画一次、改个滤波参数
重画一次、想看某个时窗的频谱还要再单独写一段——图当然能出来，但整个过程是断的，
而且每次都得重来。gathervis 把这件事变成专业处理软件那样的查看方式：数据打开就在
浏览器里，拖滑条翻炮，框一个窗就出谱，改滤波当场更新。

* **大数据秒开。** 文件以懒加载 memmap 方式打开——浏览一炮只读那一炮的字节，
  切一刀只读那一片。100 GB 的原始二进制和 1 MB 的文件打开一样快。
* **观测系统联动可视化。** 给出炮点/检波点坐标就有 Geometry 页，观测系统叠在覆盖次数图上——点任意
  炮点直接跳到它的炮集，激活炮的排列同步高亮；CMP 覆盖次数只用坐标就能算。
* **内置 QC 工具。** wiggle / 变密度显示与行业色标、零相位 Ormsby 滤波、
  AGC / 道均衡、画窗算谱（振幅谱 + f-k）、手动 / STA-LTA 自动初至拾取，
  可跨道集保持的顶/底切除，以及手工 CMP 速度分析。
* **cigvis 风格的三维视图。** 数据体可渲染为可旋转的长方体 + 三个实时切片平面
  （WebGL），滑条或键盘都能移动切片。
* **干净专注的界面。** 侧栏按显示、处理、工具分组，只显示对当前画面起作用的控件；
  翻炮、移切片都能用键盘。页面不从互联网加载任何资源，离线集群或防火墙后面打开
  一样快。
* **简单好用。** 只要一个浏览器：单端口服务，VS Code Remote-SSH 自动转发端口，
  Jupyter 里直接内嵌渲染，裸 `ssh -L` 也行。四个依赖，pip 装完就能用。

## 安装

```bash
pip install -e .                  # numpy + panel + bokeh + plotly
pip install -e ".[segy]"          # 可选，SEG-Y 导入
pip install -e ".[examples]"      # 可选，torch + deepwave，跑正演示例用，约 2.5 GB
```

用 uv 的话 extra 名字一样：

```bash
uv sync                           # 查看器本体
uv sync --extra examples          # 加上 torch/deepwave，跑 examples/deepwave_*.py
uv run --extra examples examples/deepwave_line2d.py    # 或者只为这一次运行装
```

`pyproject.toml` 里把 torch 钉在了 cu121 索引（本地开发用）。没有 GPU、或者想
少下点东西，给 uv 指 CPU 索引：
`--index https://download.pytorch.org/whl/cpu`。只有那两个 `deepwave_*` 示例
需要这些，其余一切都只靠四个基础依赖。

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
`python examples/velocity_picking.py` 在一个已知答案的 CMP 道集上打开速度分析；
`python examples/velocity_analysis.py` 在一个答案已知的模型上跑完整条速度分析链，
并把拾取结果与真值逐层对照打印（第一个参数给 `report` 则不开浏览器）；
`python examples/deepwave_line2d.py` 用 [deepwave](https://github.com/ar4/deepwave)
正演一条二维测线；`python examples/deepwave_lines3d.py` 在一个丰富的三维速度模型
（倾斜层 + 背斜穹隆 + 低速河道透镜体）上用 deepwave 的三维引擎**打五条二维测线**，
跑完直接打开查看。

## 功能

### 炮集浏览与观测系统

**Shot gathers** 页：图上方的工具栏放着炮号滑条（拖动过程中实时更新）、可直接
输入炮号的框，三维炮集还有排序选择；**←** / **→** 逐炮步进，按住 **Shift** 一次
10 炮，**Home** / **End** 跳到首炮和末炮。视图按钮在同一行的右端，全屏时也够得着。

侧栏按看数据的顺序分组：**Display**（density 或 wiggle，然后只显示该模式用得到的：
density 有色标和分辨率，wiggle 有道数；再是 clip 和极性）、**Processing**（滤波与
增益，各自只在开启时显示参数）、**Tools**（*Analysis windows*、*Spectrum*、
*Event picking*、*First breaks*、*Mute* 为可折叠分组，最后一组是手势与按键说明）。
有专属设置的页签在显示时再加一组：Geometry 页是图层和 CMP 分箱，体切片页是轴向拉伸。

**Geometry** 页是一张图：底下是 CMP 覆盖次数，上面叠炮点和检波点，当前炮以星标
表示并高亮其排列。点击任一炮点即选中该炮并切回 Shot gathers 页；侧栏的 *Map layers*
可分别开关覆盖次数、检波点、炮点三个图层。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_geometry.png" width="820" alt="Geometry 页：五条二维测线叠在 CMP 覆盖次数图上，当前炮以星标表示并高亮其排列"/>

### 三维炮集的道序排列

一个三维炮集是若干条接收线组成的 patch。处理流程中通常把整个 patch 放在同一张
`(道, 时间)` 面板上，改变的是道的排列顺序；Shot gathers 页的 **Sort** 选择器提供
四种顺序。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/sorts.png" width="820" alt="同一个三维炮集分别按采集道序、偏移距、方位角排列"/>

*同一炮的三种排列：采集道序（虚线为接收线分界）、偏移距、方位角。*

| 排序 | 说明 |
| --- | --- |
| **as recorded** `(线号, 桩号)` | 采集道序，接收线首尾相接，显示为 N 条嵌套双曲线，顶点随 crossline 距离递进。用于检查几何错误、线序倒置、极性反转与死道。虚线标出接收线分界。 |
| **receiver line** | 一次显示一条接收线，即一张二维炮记录。 |
| **offset** | 按源检距排列，整个 patch 归并为一条双曲线。用于动校、切除设计与速度分析。 |
| **azimuth** | 按源到检波点的方位角排列（测量惯例，北 = 0）。用于方位特性与方位覆盖检查。 |

四种排列都是同一张二维面板，因此分析窗、频谱、拾取与全分辨率导出的用法一致。
拾取按**道**存储而非按屏幕列，换排序时跟随道走；切到单条线时线外的拾取隐藏而
不删除，切回后恢复。

`offset` 与 `azimuth` 需要观测系统；`receiver line` 需要四维
`(shot, recy, recx, time)` 数据，线结构在该形状下才是显式的。选择器只列出当前
数据支持的选项，二维测线且无观测系统时不显示。**Shot volume** 页同时把当前炮
渲染为长方体，使用同一条滤波/增益链。

示例：`python examples/quickstart.py patch`。

### CMP 覆盖次数图

在 **Geometry** 页观测系统图的下层——同一套观测系统问两遍，覆盖上的空洞就是
正压在上面的排列缺口。对每个炮检对的中点做矩形装箱并计数，网格以第一个中点为
中心对齐，规则分布的中点落在面元正中而不是面元边界上。侧栏的 *CMP bins* 设置
面元大小，并列出网格、活动面元数、覆盖次数范围和总道数。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/fold.png" width="520" alt="正交陆上三维的 CMP 覆盖次数图，中间满覆盖、边缘羽化"/>

*正交陆上三维的覆盖次数：200 m 接收线上 25 m 桩距，200 m 炮线上 50 m 炮点，
炮线与检波桩号错开半个桩距，固定满排列。*

bin 尺寸默认取 in-line 半个桩距、cross-line 半个线距，即自然的 CMP 采样；两个
方向都可在侧栏 *CMP bins* 里修改，*Default bin size* 恢复默认。覆盖次数用 viridis
色标绘制，空 bin 以背景色绘制，不占用色标最低端。面元设置下方列出网格、有效 bin
数、覆盖次数范围与平均值，以及道总数——后者应等于 `炮数 × 检波点数`，可作几何校验。

计算只用坐标，不读取道数据，因此代价与数据体积无关，对没有道数据的裸
`Geometry` 同样可用：

```python
from gathervis.survey import fold, default_bin
grid = fold(ds.geometry)                     # 或 fold(geo, (12.5, 100.0))
grid.counts                                  # (ny, nx) 每个 bin 的道数
grid.nlive, grid.extent, grid.bin_of(x, y)
```

点击某个 bin 会选中并勾出它。背后的道索引——哪一炮的哪一道成像了这个
bin——是 `traces_in_bin`；该 bin 的偏移距-
方位分布出自同一模块，不过应用内的玫瑰图尚未实现：

```python
from gathervis.survey import (traces_in_bin, offsets_azimuths_in_bin,
                              azimuth_sectors)
shots, recs, offs = traces_in_bin(ds.geometry, grid, ix, iy)   # 按偏移距排序
offs, azis, shots = offsets_azimuths_in_bin(ds.geometry, grid, ix, iy)
edges, counts = azimuth_sectors(azis, nsector=24)   # 测量惯例，北 = 0
```

`azimuth_sectors` 返回的非空扇区数即该 bin 的方位覆盖情况。

### 顶切除与底切除

道集面板上两个点工具，一条线一个：点击加节点，拖动移动，BACKSPACE 删除选中的。
切除线画在**每一道**上，所以你看到的就是实际会被应用的那条线，而不只是你放下的
几个手柄。打开 **Apply mute** 看切除后的道集——看不到它切掉了什么，就判断不了
线画得对不对。

**线存的是偏移距，不是你放节点时那一列的列号。** 切除本来就是偏移距的函数，
存偏移距，换到下一炮（道数不同、顺序不同）这条线依然是同一个切除，重排序面板
也不会错位。没有观测系统时退回道号，这是诚实的但就没法在道集之间通用了。

**一条线管所有，允许例外。** *Edits go to* 决定一次编辑写到哪里——*All gathers*
改的是所有道集继承的默认线，*This gather* 给当前这一个单独一条。在某一道集上
调好了想推给全部，用 *Make default*；想撤掉某个例外，用 *Revert*。Mute 分组底部
的状态行随时告诉你当前用的是哪一条。道集之间不做任何插值：
一个道集要么有自己的线，要么用默认线。

导出的是 JSON，同一个文件在脚本里就能切除道集，不需要开界面：

```python
from gathervis.mute import Mutes, mute_shot

mutes = Mutes.from_json(open("mutes.json").read())
clean = mute_shot(ds, 7, mutes)        # 第 7 炮若有自己的线就用它自己的
```

底层部件对任意道集和它的偏移距都能用：

```python
from gathervis import mute

top = mute.linear_line(v=1500., x_max=3000., pad=0.05)   # 一条直线切除
out = mute.apply(gather, offsets, dt, top=top, taper=0.04)
mute.evaluate(top, offsets)                              # 每道的切除时刻
```

缓坡不是装饰。硬切会沿着切除轨迹留下一道阶跃，而阶跃是走在相干路径上的宽带
能量——它会在谱和叠加里以一个地下根本没有的同相轴的形式回来。

### 速度分析

```python
from gathervis import cmp, survey
import gathervis as gv

grid = survey.fold(ds.geometry)
cg = cmp.gather_in_bin(ds, grid, ix, iy)
gv.velocity_analysis(cg, dt=ds.dt, port=8080)
```

三个面板从左到右，也就是干活的顺序：CMP 道集、它的相似度速度谱、按当前拾取
做完动校正的道集——再加上这一切叠加出来的那一道。

拾取靠手。在速度谱上点击加一个 `(v, t)` 点，拖动调整，BACKSPACE 删除。
**一边拖一边看动校正面板**：速度偏低，远偏移距端同相轴会上翘；偏高则下弯；
对了就是平的。这是判断拾取对错的唯一办法，也是这两个面板并排放的理由。
**没有自动拾取，这是刻意的**——查看器应该把答案摆给你看，而不是替你决定。

*export velocity file* 导出 `time_s,velocity`，头部带道集位置，也能读回来。

入口吃的是**传进来的**道集，不是面板自己去找的：`CmpGather`（自带偏移距和
bin 位置）、SEG-Y 排序结果，或者任意 `(ntrace, nt)` 数组加上 `offsets=` 和
`dt=`。

`python examples/velocity_picking.py` 在一个已知答案的合成数据上打开它，并把
答案打印出来，方便你对照自己拾的。

底层部件也能单独用：

```python
from gathervis import velocity as vel

clean = vel.front_mute(cg.data, cg.offsets, ds.dt, v=1500., pad=0.05)
spec, v_grid = vel.velocity_spectrum(clean, cg.offsets, ds.dt, nv=100)
flat = vel.nmo(clean, cg.offsets, v_t, ds.dt)
trace = vel.nmo_stack(clean, cg.offsets, v_t, ds.dt)
v_int = vel.dix(v_t, ds.times, smooth=51)       # 层速度
```

速度谱返回 `(nv, nt)`——速度坐在道号平时的位置上，时间和这里每一个面板一样放
在最后一维。

### 自己组装查看器

`gv.show(data)` 会挑出数据支持的视图并排好。如果这不是你要的——想少几个 tab、
换个顺序、或者只要一个面板——就自己搭：

```python
import gathervis as gv

s = gv.session(data)
s.add(gv.gather())        # 炮集浏览
s.add(gv.geometry())      # 观测系统图
s.show(port=8080)
```

或者一次写完，给出的顺序就是 tab 顺序：

```python
gv.session(data, views=[gv.geometry(), gv.gather()]).show(port=8080)
```

可用的视图是 `gather`、`geometry`、`shot_volume`、`slices`，而
`gv.show()` 就是把这个列表替你填好的同一件事。显式要一个数据支持不了的视图
（没有坐标却要观测系统图）会直接报错，而不是悄悄少一个 tab。

两层共享其余一切：显示控件、侧栏、URL 同步、键盘。选择状态也共享，走的是
session 的游标而不是视图之间直连——在图上点炮点和拖动炮号滑块是同一件事的两种
说法，每个视图都能听到，而它们互相并不知道对方存在。

### 操作速查

| 想做什么 | 怎么做 |
| --- | --- |
| 找回完整画面 | 工具栏行里的"重置视图"按钮。视图本身就被锁在数据范围内，根本拖不出去 |
| 画矩形窗 | 工具栏选中框选工具 → **SHIFT+拖拽**（或点一下、移动、再点一下） |
| 画多边形窗 | 工具栏选中多边形工具 → 逐点点击加顶点，**双击或 ESC** 收尾 |
| 移动 / 删除窗 | 直接拖动；点选后按 **BACKSPACE**（或点 *Clear windows*） |
| 加 / 移 / 删拾取点 | 工具栏点拾取工具 → 点击添加、拖动移动、点选 + **BACKSPACE** 删除（或 *Clear shot* / *Clear all*） |
| 自动拾初至 | *First breaks* 里设好 STA / LTA / Trigger ratio → *Auto pick*；再用 *Event picking* 里的 *Snap*（peak / trough / \|max\|）或手动拖动精修 |
| 比较谱峰高度 | 工具栏十字线图标开准星，配合面板下方的 freq / dB 读数 |
| 计算 f-k 谱 | Spectrum 分组的 *f-k of first window*，作用于第一个窗 |
| 复用一组分析窗 | Analysis windows 分组的 *Export* / *Import*（JSON） |
| 拾取结果导出再导回 | Event picking 分组的 *Export* / *Import*（CSV），格式 `shot,trace,time_s` |
| 单轴伸缩 | 工具栏的 x-only / y-only 滚轮缩放工具 |
| 调整三维长方体比例 | 侧栏 *Volume view* 里每个轴一条拉伸滑条（×1 居中，×0.125–×8）；*Reset camera* 恢复视角 |
| 移动体切片 | **1** / **2** / **3** 选切片，**←** / **→** 移动，按住 **Shift** 大步走，**Home** / **End** 到两端；当前切片有描边 |
| 跳到某一炮 | 拖炮号滑条、在旁边的框里输入炮号，或在 Geometry 页点炮点 |
| 重排三维炮集 | 工具栏行里的 *Sort* 选择器（采集道序 / 接收线 / 偏移距 / 方位角） |
| 重新划分覆盖网格 | 侧栏 CMP bins 的 *Bin x* / *Bin y*（*Default bin size* 恢复默认） |
| 选中一个 CMP bin | 在 Geometry 页的图上点它（避开炮点） |
| 只看部分图层 | 侧栏 *Map layers*：覆盖次数、检波点、炮点 |
| 拾取速度 | 速度谱上用点工具 → 点击添加，拖动移动，点选 + **BACKSPACE** 删除 |
| 判断速度拾取对不对 | 看动校正面板——上翘是偏低，下弯是偏高，平了才对 |
| 逐条翻接收线 | *Sort* 选为 receiver line，再拖旁边的 *Line* 滑条 |
| 调谱面板大小 | Spectrum 分组里的 *Panel size*（S / M / L） |
| 存图 | **Full image** 按钮（原分辨率全图），或工具栏 save 图标（当前视野，屏幕分辨率） |
| 翻炮 | **←** / **→**，按住 **Shift** 一次 10 炮，**Home** / **End** 到首末炮 |
| 分享当前视图 | 复制地址栏：当前页签、炮号、排序、线号、色标、clip、极性、滤波、增益（含 AGC 窗长）都写在 URL 里 |

同一份速查在应用里 Tools 的 *Gestures and keys* 分组可看。只占用在数据里移动的键：
方向键、Home / End，以及体切片页上的 1–3；其余控件设一次即保持。光标在输入框内
时按键不响应，`gv.show(..., keys=False)` 可整体关闭。

每个浏览器标签页各有独立会话：同一服务开两个标签页互不影响，刷新页面则按 URL 恢复状态。

### 三维体切片

任何三维数组——整条测线、一个三维炮集、一个速度模型——渲染为线框长方体加三个
轴对齐切片平面。视图上方的滑条移动切片，每步只传输该切片的抽稀数据；键盘也可以：
**1** / **2** / **3** 选中切片（选中的切片有描边），方向键移动。旋转与缩放后相机在
后续更新中保持；每个轴各有一条拉伸滑条（×0.125–×8，几何级数档位，×1 居中）。
页首第二张图把炮切片推到中间，0.5 s 的时间切片因此露出在长方体内部。

整条测线的体切片页在第一次打开之前不读任何数据，打开时显示加载提示：叠前数据按道
存储，一张时间切片要从它显示的每一道各取一个样点，在几十 GB 的文件上几乎等于把
文件读一遍。切片始终按显示分辨率读取，不会先读全分辨率再抽稀。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/velocity_qc.png" width="440" alt="depth 轴速度模型 QC"/>

上图为同一套正演所用的三维速度模型，以 `axes=('x','y','depth')` 查看；属性体
自动使用深度标注与 min/max 色标范围。

本页只提供一个长方体和三个切片面，用于快速查看道集所在的数据体，不是体解释
工具。**数据本身是数据体时，推荐使用
[cigvis](https://github.com/JintaoLee-Roger/cigvis)**：它可将三维地震数据与标签、
断层、RGT、层位面、井轨迹与测井曲线、三维地质体一并可视化，也支持二维与一维
数据；三维后端为 vispy，二维/一维为 matplotlib，Jupyter 环境为 plotly，另有
viser 与 OpenVDS 可选。gathervis 的切片视图沿用其交互风格。

### 显示方式与色标

所有二维面板可在变密度与变面积 wiggle 之间切换，侧栏只显示当前模式的设置：
density 有色标和分辨率，wiggle 有道数。*Flip polarity* 翻转显示极性，wiggle 充填瓣
与变密度色标同步互换（SEG normal ↔ reverse）。色标为 `gray`（默认）、`seismic`、
`petrel`（锚点取自 cigvis）、`rainbow`、`viridis`，各有 `_r` 反转变体。*Clip* 滑条
（剪切百分位）设定色标范围，在 wiggle 模式下同时作为增益。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/display_modes.png" width="620" alt="变密度 vs wiggle"/>

### 滤波与增益

梯形（Ormsby 式）**低通 / 高通 / 带通**滤波：`f1–f2` 低切斜坡、`f3–f4` 高切斜坡，
线性过渡，零相位。增益为 **AGC**（滑动窗 RMS，窗长可调）或**道均衡**。处理链为
滤波 → 增益；增益开启时色标范围从处理后的道集重新计算。

处理按当前显示的道集逐炮进行，不影响 memmap 懒加载。Volume slices 页使用同一条
链：512 MB 以内的数据体在链变化时整体处理一次，保证三个切片一致；超过该尺寸的
memmap 回退为原始数据，并在侧栏提示。输入为 CuPy 数组时，滤波、增益与拾取在
GPU 上执行。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/filters.png" width="620" alt="带通滤波压制含噪道集"/>

### 分析窗与振幅谱

在道集上画**矩形**窗（SHIFT+拖拽，或点-移-点）或**多边形**窗（逐点点击，ESC
收尾），点 *Compute spectrum* 计算窗内数据的振幅谱：对门控后的样点直接做 `rfft`，
绘制 `|X|` 相对本窗峰值的值。不加窗函数，不做归一化与平滑。

*Traces* 决定窗内各道如何进图：**Per trace**（逐道绘制）、**Mean**（幅值平均）、
**Middle trace**（取窗中心道）。*Scale* 在 **Linear** 线性振幅（归一到最大曲线峰值
为 1.0）与 **dB** 之间切换。图例标明绘制内容与频率间隔 `df = 1/时窗长度`。

不加窗函数带来两个后果，均为矩形截断后 DFT 的固有性质：矩形门的旁瓣按 1/f
衰减，低于峰值约 40 dB 之后读到的主要是窗函数而非数据；单道周期图是 2 自由度
估计，其抖动不随记录长度减小。多道平均可将抖动降低 `sqrt(N)` 倍，但仅对相互
独立的道成立——相干道集上各道近似相同，平均值与单道差别很小。加窗函数等价于对
复谱做卷积（周期 Hann 即 `(-1/4, 1/2, -1/4)` 三点卷积），属于平滑，故不采用。

窗定义可导出与导入 JSON，内容为道/时间坐标与 dt。

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/windows_spectra.png" width="720" alt="分析窗及其谱"/>

### f-k 谱

*Spectrum* 分组的 **f-k of first window** 对**第一个**已画窗做二维变换：横轴波数，纵轴
频率，幅值为相对本窗峰值的 dB，动态范围固定 60 dB。线性同相轴在 f-k 域中表现为
过原点的射线，视速度不同的事件因此分离，可用于判断是否需要以及如何设计视速度
滤波。

约定：**k 为正表示同相轴向道号增大的方向倾斜**。横轴单位为周/道而非周/米——道间距
不一定均匀，按偏移距或方位角排序后尤其如此。与一维谱相同，计算基于当前显示的
（即经过滤波/增益链的）道集，换炮、换排序或改滤波后自动重算。窗至少需覆盖 4 道
与 8 个采样点。两个谱面板共用 *Panel size*（S / M / L），均带十字线与 `k` / `f` 读数。

### 分辨率：到底有多少真的送到了浏览器

侧栏 Display 分组里的两个控件决定画出多少道集，各自只在适用的模式下出现，而且
在做抽稀时都会在面板下方明说。

**Resolution**（密度模式）。*Auto* 把图像限制在每轴 1600 像素——在 192 × 1250
的道集上这个上限根本不生效，一点没抽。*Full* 则把每一道、每一个采样点都送过去。
抽稀发生在**像素**层面：两种情况下面板都覆盖整个道集、坐标轴也不变，所以 *Auto*
让你损失的是细节，不是范围。

**Traces**（wiggle 模式）。wiggle 显示画多少道。想密就调大，想干净就调小。和密度模式不同，
这个是真的丢整道，所以只要没画全就会提示。

默认值是预算而不是决定，因为代价是实打实的：

| 道集 | auto | full |
| --- | --- | --- |
| 192 x 1250 | 0.2 MB, 0.7 ms | 0.2 MB, 0.5 ms（上限没触发） |
| 960 x 3000 | 1.4 MB, 3.4 ms | 2.9 MB, 3.6 ms |
| 2400 x 4000 | 1.6 MB, 4.6 ms | 9.6 MB, 24 ms |
| 5000 x 12000 | 1.9 MB, 7.3 ms | 60 MB, 200 ms |

这是**每次重绘**的代价，而每换一炮就重绘一次。几千道以内 `full` 基本免费，再往上
就是一个值得自己拿主意的取舍——所以它是个控件，不是一个常数。

**Full image** 不受这两个控件影响，它始终按每道每采样一个像素重新渲染。

### 全分辨率导出

面板上屏前按像素预算做 stride 抽稀，bokeh 工具栏的 save 存的是当前 canvas 截图，
两者都不是数据自身的分辨率。工具栏行最右端（fit、fullscreen 旁）的 **Full image** 按钮按
一道一像素、一采样一像素重新渲染并下载无损 PNG，色标、显示方式、剪切百分位、
极性、滤波与增益均与屏幕一致；wiggle 模式下绘制全部道，而非浏览器收到的子集。
PNG 由 numpy 与标准库 zlib 写出，不依赖 matplotlib 或无头浏览器。

需要当前视野的局部截图时用工具栏的 save 图标。

脚本中批量出图：

```python
from gathervis.process import bandpass, agc
from gathervis.render import save_png

a = agc(bandpass(g.shot(7), g.dt, f3=60, f4=80), g.dt)
save_png("shot007.png", a, cmap="gray")            # 或 display="wiggle"
```

### 初至与同相轴拾取

绘图工具栏的 *pick* 工具：点击添加拾取、拖动移动、BACKSPACE 删除选中项。拾取点
按道号以虚线相连。*First breaks* 分组的 **Auto pick** 对当前显示（即滤波后）的
道集运行 gapped STA/LTA 初至拾取，STA、LTA 与触发比（trigger ratio）可调；LTA 窗在 STA 窗起点结束，因此靠近记录开头的初至也能
触发，且触发需持续约 STA/2 以排除单点噪声。*Event picking* 里的 **Snap** 将拾取吸附到所在道最近的
波峰 / 波谷 / |最大值|，搜索范围 ±30 ms。

拾取按炮存储并随浏览跟随，键为道号而非屏幕列，因此换排序后仍对应同一道。可导出
与导入 CSV，格式为 `shot,trace,time_s`。

### SEG-Y 导入

`gv.from_segy('field.sgy')`，或命令行 `gathervis field.sgy`。dt 读自二进制卷头，
道按 FFID 分组为炮，炮点与检波点坐标读自标准道头字并按 SEG-Y 标量规则换算，
Geometry 页（含覆盖次数图）随即可用。道数不等的炮补零对齐并打印警告。需要可选依赖
`segyio`；文件整体读入内存，估算超过 `max_gb`（默认 8 GB）时拒绝打开。

## 远程使用

1. **VS Code Remote-SSH（推荐）。** 在集成终端里跑任意示例；VS Code 自动转发
   端口，打印出的 `http://localhost:8080` 直接可点。零配置。
2. **Jupyter。** 省略 `port=`，返回的 app 直接在 notebook 里内嵌渲染——
   完全不需要额外端口。
3. **裸终端。** `ssh -L 8080:localhost:8080 user@gpu-server`，然后本地打开
   URL。端口被占用时 gathervis 自动换一个空闲端口并打印出来。

页面不从互联网加载任何东西：没有网络字体，没有 CDN 脚本或图标。在离线集群上、
或者在访问这些站点很慢甚至被屏蔽的网络里，打开速度和别处一样。每个浏览器标签页
各是一个独立会话，同一台服务器上的同事互不影响对方的炮号，分享出去的链接打开的
就是链接里描述的那个视图。

## 数组语义

最后一维永远是纵轴：数据用 **time**，属性体用 **depth**。裸三维数组默认为
`(shot, rec, time)` 并逐炮浏览；四维数组默认为 `(shot, recy, recx, time)`，每炮
一个三维炮记录。传 `view='slices'` 可把三维数组改为体切片查看，或用 `axes=`
显式声明其他语义。`axes` 说明数组*是什么*，`view` 说明*怎么看*，`sort`（在应用里）
说明*按什么顺序排*——三者独立。

## 设计笔记

* **懒加载。** 数据以 memmap 打开；浏览一炮只读该炮的字节，切片只读该切片；
  滤波与谱分析只作用于当前显示的道集。按采集道序或接收线排序也是懒的（一次
  reshape、一次切片），只有按偏移距/方位角排序需要实体化该炮的数据——一炮，
  不是整个文件。
* **传输格式。** 面板在服务端按像素预算跨步抽稀并量化为 uint8 后再发往浏览器。
* **排序与处理分离。** 排序只改变道的顺序，返回的仍是该炮自己的道，滤波/增益链
  作用于排序结果，因此二维与三维炮集共用一条处理路径。
* **两种渲染方式。** 二维面板为 bokeh 图像 / wiggle，数据体为 plotly WebGL 切片
  平面，二者共用同一条抽稀 → uint8 管线。
* **导出路径独立。** 导出图不经过抽稀：同一个处理后的数组按原分辨率光栅化，由
  numpy 与标准库 zlib 编码为 PNG，因此不引入额外的绘图或图像依赖。
* **外观集中在一处。** `gathervis/theme.py` 放着设计系统（一个 Panel design 加
  一份作用于所有控件的样式表）、页面模板和图件样式；viewer 代码只管画面上有什么，
  theme 管它长什么样。系统字体，唯一的强调色只用来表示状态。
* **每个浏览器标签页一个会话。** bokeh 模型只能属于一个文档，所以每个会话都在同一份
  （惰性加载的）数据上建自己的 workspace，而不是共用图件。
* **交互状态保持。** 滑条拖动过程中更新；三维相机、拉伸倍率与已画的窗在更新后
  保留。每个二维面板带 x / y 单轴滚轮缩放和光标读数（道号、时间、当前像素振幅，
  纯客户端计算），谱面板带同样的读数与可开关的十字线。

## 路线图

**三维采集质控**（剩余）：炮记录时间切片面积图 · 逐 bin 的偏移距-方位玫瑰图
（三维炮集排序、CMP 覆盖次数图：已完成）。
**M2**（剩余）：联动对比面板 · 客户端体缓存（cigvis 级切片拖动手感）
（AGC / 道均衡 / CuPy：已完成）。**M3**（剩余）：道头索引、按任意键抽道集
（SEG-Y 导入：已完成）。**M4**（剩余）：共偏移距/时间切片
（同相轴拾取、NMO 预览、速度谱、手工速度分析：已完成）。完整规划见 [`docs/plan.md`](https://github.com/zzzzswh/gathervis/blob/main/docs/plan.md)。
*以及任何你提出的需求——见页首。*

## 致谢

速度分析这套方法是 2025 年在成都 SEG 的一次课程里 **Gerard Schuster 教授**
教给我的。非常感谢他。基于它做的自动 K-means 拾取在
[kmeans-vel-picker](https://github.com/zzzzswh/kmeans-vel-picker) 里；这个
仓库里是手工拾取，因为下判断是人的事。

`petrel` 色标锚点取自 [cigvis](https://github.com/JintaoLee-Roger/cigvis)（MIT）。
正演示例使用
[deepwave](https://github.com/ar4/deepwave)。基于
[Panel](https://panel.holoviz.org/)、[Bokeh](https://bokeh.org/) 与
[Plotly](https://plotly.com/javascript/) 构建。

## 许可证

MIT
