# 基于 HotKnots 2.0 的 CPLfold 独立伪结能量计算

> **来源与“独立”的含义：** 本实现是基于 HotKnots 2.0 能量计算源码和
> 参数表完成的 Python 移植与重构，不是与 HotKnots 无关的 clean-room 实现，
> 也没有重新拟合 DP/CC/RE 参数。本文的“独立”仅表示 CPLfold **运行时**
> 不需要 HotKnots 的包装器、源码树、共享库或原生可执行文件。

## 1. 与 HotKnots 的关系

重构前的 CPLfold 通过当时仓库中的 `Utils/HotKnots_v2.0/hotknots.py` 启动
`bin/computeEnergy`。它只调用 `compute_energy()` 给一个已知结构打分，完全没有调用
HotKnots 的 hotspot 生成、启发式搜索、候选结构扩展或绘图代码。因此本次
工作不是移植整个 HotKnots 预测器，而是从其源码和参数中移植、重构 CPLfold
需要的能量计算部分：

```text
CPLfold.py
  -> compute_energy(sequence, structure, model)
     -> dot-bracket 配对表和 loop/band 分解
     -> SimFold FM363 普通二级结构能量
     -> DP / CC / RE 伪结修正
     -> exterior dangling-end 修正
```

具体来源与处理如下：

| CPLfold 部分 | HotKnots 2.0 来源及处理 |
|---|---|
| closed-region/Loop/Bands 结构分解 | 基于 `LE/Stack.cpp`、`Loop.cpp`、`LoopList.cpp` 和 `Bands.cpp` 的逻辑移植并重构为 Python |
| 普通二级结构能量 | 基于 HotKnots 捆绑的 SimFold FM363 计分代码和参数 |
| DP/CC/RE 伪结能量 | 基于 HotKnots `Loop`/`LoopList`/`paramsPK` 的能量计算语义 |
| 八个参数文件 | 来自 HotKnots 2.0 `bin/params`；仅规范化了部分换行和尾随空格，数值不变 |
| 候选结构生成 | 仍由 CPLfold 的两阶段 LinearFold 流程完成，不使用 HotKnots 的 hotspot/启发式搜索 |

当前 `Utils/hotknots_energy.py` 从 `Utils/energy_params` 读取上述参数子集并在
Python 中完成计算。CPLfold 的默认路径不导入 HotKnots 包、不读取其源码目录，
也不启动原生可执行文件。迁移前捆绑的 `computeEnergy` 是 AArch64 ELF；
Python 版也消除了运行机器与二进制架构必须一致的
问题。完成差分验证后，整个 `Utils/HotKnots_v2.0` 已从本分支删除；这个
删除只是去除不必要的运行时组件，不改变上述代码和参数来源。

## 2. 完整 closed-region / Loop / Bands 树

只按括号类型寻找两条 stem 只能覆盖简单 H 型。完整实现移植并重构了 HotKnots
原 `Stack`、`Loop` 和 `Bands` 的结构语义：

1. 从左到右扫描所有成对端点，形成最小 closed region；交叉 pair 会被合并到同一
   region，完整嵌套的 region 则保持父子关系。
2. 对每个 region 去掉直接子 region 区间，得到当前层的 paired surface。
3. 在 surface 上按相邻配对端点连接 maximal band；一个 band 可以包含连续 stack、
   bulge 或 internal loop。
4. 根据子 region 是否落在 band arm 内，将其标为 `in_band` 或 `un_band`，并构造
   band 相邻 pair 之间的 stack/interior/multiloop span。
5. 递归计分普通 loop、pseudoloop、跨 band multiloop 以及嵌套 pseudoknot，最后按
   原 `EnergyDangling` 规则补 exterior/pseudoknot dangling ends。

因此结构树不再局限于两个 band，也无需为 kissing、链式或嵌套伪结调用 C++。

## 3. 公共的 FM363 能量

`Utils/energy_params` 中的八个表来自 HotKnots 2.0 发布包的 `bin/params`，
本工作没有重新拟合其数值。DP、CC 使用各自参数文件的前 363 项；RE 使用
`turner_parameters_fm363_constrdangles.txt`。这些参数重建了 SimFold 的：

- 21 个对称 stack 参数；
- 96 个 hairpin terminal-mismatch 参数；
- 1x1、1x2/2x1、2x2 和一般 internal-loop 参数；
- top/bottom dangling ends；
- hairpin、bulge、internal-loop 的长度惩罚；
- terminal AU/GU、特殊 hairpin、multiloop 和 tetraloop 参数。

对无伪结结构，Python 版构造普通 loop tree，并逐节点计算：

```text
Esecondary = Σ Estack/hairpin/internal/bulge/multiloop
             + Eterminal-AU + Edangling
```

大环沿用 HotKnots/SimFold 的 `1.079 ln(n/nmax)` 外推；不对称 internal loop 使用
`min(3.0, 0.5 |n1-n2|)`；GAIL 规则保持开启。

一个容易遗漏的兼容细节是：原 SimFold 将参数保存为 0.01 kcal/mol 的整数，并用
C++ 浮点乘 100 后直接截断。Python 版复现了这个历史舍入行为，否则某些 stack 会有
0.01–0.02 kcal/mol 的偏差。

## 4. DP09（Dirks & Pierce 2009）

DP09 文件由 `363 + 14 = 377` 个参数组成。对一个含 `B` 个 band 的 pseudoloop：

```text
EDP = Σband scale(loop) · EFM363(loop)
      + Pinit + B · Pb + Pup · Upk + Pps · Nunband
      + Espanning-multiloop
      + Eterminal-AU + Edangling
```

其中普通 stack 乘 `stP`，带 bulge/internal loop 的 band 项乘 `intP`。为精确兼容
原实现，DP 根据 5' 臂相邻性 `ap == a + 1` 选择缩放项；因此只有 3' 臂有 bulge 时
仍使用 `stP`。`Pinit` 根据树位置选择：顶层 pseudoloop 用 `Ps`，嵌在普通
multiloop 或 band 内用 `Psm`，嵌在 pseudoloop gap 中用 `Psp`。`Nunband` 是直接
嵌在 gap 中的 closed-region 数。DP09 参数为：

| 参数 | 值 | 含义 |
|---|---:|---|
| `Ps` | -1.38 | 启动一个 pseudoloop |
| `Psm` | 10.07 | pseudoloop 中 multiloop 修正 |
| `Psp` | 15.00 | 嵌套 pseudoknot 修正 |
| `Pb` | 2.46 | 每个 band 的代价 |
| `Pup` | 0.06 | pseudoknot loop 中每个未配对碱基 |
| `Pps` | 0.96 | 嵌套非 band 子结构 |
| `stP` | 0.89 | band stack 缩放 |
| `intP` | 0.74 | band internal-loop 缩放 |

其余 6 项 `a,b,c,a_p,b_p,c_p` 是 multiloop 经验项。尤其 band 内相邻 pair 之间
跨越多个子 region 时使用 `a_p + b_p(Bbranch+2) + c_p U`；Python 版同时复现该
multiloop 的 AU 与 dangling 项。

## 5. CC09（Cao & Chen 2009）

CC09 文件共 923 项：

```text
363 FM363 + 14 DP fallback + 546 CC
```

546 个 CC 参数包括 36 个 flush coaxial、96 个 mismatch 第一项、156 个 mismatch
第二项、两组短环 entropy 表和两组长环拟合系数。可由 CC 表处理的 H 型伪结使用：

```text
ECC = Σband EFM363
      + kBT ln(9)
      + ΔGL1(stem2, loop1)
      + ΔGL2(stem1, loop2)
      + Ecoax
      + Eterminal-AU + Edangling
```

在 37 °C 下 `kBT ≈ 0.616268 kcal/mol`。环长不超过 12 时直接查表；更长时使用：

```text
ln Ωcoil   = 2.14 L + 0.10
ln Ωfolded = a ln(L - Lmin + 1) + b(L - Lmin + 1) + c
ΔGL        = kBT (ln Ωcoil - ln Ωfolded)
```

中央环为 0 或 1 nt 时还会分别计算 flush 或 mismatch coaxial stacking。这里不是
只要 coaxial 参数为负就直接采用：原 HotKnots 会比较 coaxial stacking 和占据同一
junction 的 dangling ends，只有 coaxial 严格更有利时才采用，并阻止这些碱基再次
作为 dangle 计分；若相邻的嵌套 branch 共享该碱基，则还按
`simple_dangling_ends=1` 使用 branch 一侧的 dangle 参数。Python 版复现了这些选择
与占用规则。以下情况与 HotKnots 一样自动退回参数文件中携带的 DP 模型：stem 长度
不在 2–12、需要的短环
表项缺失、loop1/loop2 为 0，或拓扑不适合 CC 表。多于两个 band、kissing/chain、
band span 中有 multiloop 等情况都走该明确定义的回退路径，并不是未计分或近似跳过。
stem 范围会在访问 2–12 行的 entropy 表或长环公式前检查，越界结构不会因表下标而
失败。

## 6. RE（Rivas & Eddy）

RE 使用 Turner/SimFold 基线能量，并对 band 中的 stack/internal loop 统一乘
`g = 0.83`：

```text
ERE = 0.83 Σband EFM363
      + Gw + Gwh(B - 2)
      + Q_tilda · Uall
      + 2B · P_tilda
      + Pi · Nunband
      + Espanning-multiloop
      + Edangling
```

仓库参数为 `Gw=7`、`Gwh=6`、`Q_tilda=0.2`、`P_tilda=0.1`、`Pi=0.1`。
`Uall` 与 DP 不同：它也包含 band 内 bulge/internal-loop 的未配对碱基。原 RE 路径
还不会添加最外层或 band 端点的 terminal AU/GU penalty；Python 版保留这一行为，
而不是把 DP/CC 的规则误套到 RE。

跨 band multiloop 使用 RE 的 `M_tilda`、`p_pairedMultiPseudo` 和普通 FM363
multiloop helix/AU/dangling 项；嵌套 pseudoknot branch 按原实现计为两个 helix。

## 7. dangling-end 的历史语义

原 `computeEnergy` 的两个输出字段是：

```text
energy                  = 主能量 + exterior/pseudoknot dangling ends
energy_no_dangling      = 主能量
```

字段名不完全准确：multiloop 内部的 dangling ends 已包含在“主能量”中，因此第二列
仍保留它们。Python 版按原程序复现了这一点。H 型伪结还必须按 unpaired run 扫描，
不能给每个 stem 端点独立加 dangle；紧凑结构 `((.[[)).]]` 的两个孤立碱基实际上均
不产生 dangling energy。

## 8. 支持边界

当前 Python 抽取支持：

- 任意无 crossing pair 的普通二级结构；
- 两条 crossing stem 构成的 H 型伪结；
- 同一结构中多个彼此独立、区间不重叠的 H 型伪结；
- band 中连续 stack、bulge 和 internal loop；
- pseudoloop 内嵌套普通 stem，以及普通 loop 内嵌套 pseudoknot；
- pseudoknot 嵌在另一个 pseudoknot 的 band 或 gap 中；
- 多 band 链式结构与 kissing pseudoknot；
- 跨 band 的 multiloop；
- pseudoknot 区间内外的普通二级结构组件；
- `DP03`、`DP09`、`CC06`、`CC09`、`RE`；其中 `DP09`、`CC09`、`RE` 是本次
  重点实现和验证的模型。

输入仍须是长度一致的 RNA 序列和成对的 dot-bracket；与 HotKnots 一样，`.` 和 `_`
都表示未配对位置。所有 pair 必须是 AU、CG 或 GU canonical pair。
每个 hairpin 必须至少包含 3 个未配对碱基。原 C 程序对 0--2 nt 的非法 hairpin
返回约 `16000 kcal/mol` 的内部 `INF` 哨兵；Python API 明确抛出 `ValueError`，避免把
哨兵误当成可排序的物理能量。因此下文的数值一致性结论针对上述合法输入域。
`UnsupportedTopologyError` 仅为旧版调用方保留；上述合法复杂拓扑不再触发它，
CPLfold 也不再以“unsupported”跳过候选。

## 9. 与原程序的验证

移植验证阶段曾在 x86-64 上重新编译未改变算法的 HotKnots 2.0，并逐结构比较两列
输出。该 oracle 不属于当前分支的运行时或测试依赖：

| 覆盖项 | 比较数 | 结果 |
|---|---:|---|
| 21 类普通结构模板，随机 canonical sequence，DP09/CC09/RE | 126 | 与原输出一致 |
| 连续 H 型（长度、三段 loop、CC fallback 组合） | 30 | 一致 |
| band 内含 bulge/internal loop 的 H 型 | 15 | 一致 |
| CC09 mismatch coaxial 随机序列 | 12 | 一致 |
| README 序列的 5 个 CPLfold merged structures | 15 | 一致 |
| 外层/嵌套 GGG hairpin，DP09/CC09/RE | 6 | 一致 |
| 两个独立 H 型组件（含 CC entropy 与 fallback），DP09/CC09/RE | 6 | 一致 |
| 普通 stem 嵌在 pseudoloop、pseudoknot 嵌 band/gap | 9 个模型向量 | 一致 |
| 三 band kissing/chain、3–5 band 密集链 | 12 个模型向量 | 一致 |
| 跨 band multiloop（4 种分支布局） | 12 个模型向量 | 一致 |
| 80 个随机复杂结构，2–10 bands，三模型各比较两列 | 480 个数值 | 一致 |
| ArchiveII 测试序列上由 CPLfold 生成的 13 个假结候选，DP09/CC09/RE 各比较两列 | 78 个数值 | 一致；最大绝对误差 `2.16e-6 kcal/mol` |

ArchiveII 的输入需要作一个区分：扫描本地 `data_archiveII` 的 30 个
`train/dev/test.conllx` 文件，共有 39,500 条记录出现（3,435 条唯一序列），其参考
配对中没有 crossing pair；只扫描十个 `test.conllx` 则是 3,950 条记录，同样没有
参考假结。因此上表不是拿无假结的参考结构冒充假结测试，而是从各测试集读取真实
ArchiveII 序列，让 CPLfold 生成 crossing 候选，再把完全相同的 sequence/structure
分别交给 Python 版和临时编译的原 HotKnots `computeEnergy`。13 个去重候选来自 16S、
5S、SRP 和 tRNA，覆盖单个 H-type、两个独立 H-type，以及三 band chain/kissing
拓扑，共比较 39 个模型 case、78 个能量值（总能量和 no-dangling）。

这次数据集差分最初检出一个真实的 `0.5 kcal/mol` 偏差：DP/CC 路径漏加了跨 band
multiloop 后重新开始的 AU/GU helix-end penalty。原 `Loop::pseudoEnergyDP()` 在
`multiPseudoEnergyDP()` 之后会给该 band 内侧第一个配对加此项；Python 版现按每个
spanning multiloop 的 `inner` pair 复现，并把触发问题的 ArchiveII 16S 候选及一个
最小化案例固化为回归测试。RE 本来就不使用这项，因此未作改变。

随后又使用保留更多输出精度的临时 x86-64 oracle 做了第二轮独立差分。审计覆盖
1,839 个生成的 sequence/structure 输入（普通 loop 与 FM363 参数组合、CC flush/
mismatch coaxial、2–30 band 随机和链式结构），共 3,303 个模型 case、6,606 个
`energy`/`energy_no_dangling` 数值。它检出两个 CC09 case（偏差分别为 `0.68` 和
`0.85 kcal/mol`）：旧 Python 版看到负 coaxial 参数便直接采用，没有先与同 junction
的 dangle 比较。修复并加入最小回归案例后，这 6,606 个数值全部一致，最大绝对误差
为 `2.08e-5 kcal/mol`。再次运行上述 13 个 ArchiveII 候选的 78 个数值也全部一致。

第三轮审计改用此前未系统覆盖的边界与组合，共检查 6,202 个生成的
sequence/structure 输入、12,998 个模型 case、25,996 个能量值，包含：任意和紧凑
crossing 配对图、DP03/DP09/CC06/CC09/RE、CC stem/loop 边界、两张 entropy 表的全部
stem 行和短环列、长环至 100 nt、2,880 种 flush/mismatch coax junction，以及
pseudoloop 中嵌套 hairpin/stack/interior/multiloop/pseudoknot。它又检出并修复了三处
问题：

- CC stem 超过 12 且关联 loop 超过 12 时，在 DP fallback 前误访问长环公式，导致
  `IndexError`；
- CC coax 邻接嵌套 branch 时，`simple_dangling_ends` 分支使用了 junction 另一侧的
  错误碱基，600 个定向 case 中有 21 个产生 `0.5–2.07 kcal/mol` 偏差；
- pseudoloop 内独立计分的普通 closed region 没有隔离序列左边界，使内部 hairpin
  读取区间外的 `GGG` 并误加 `0.05 kcal/mol` bonus。

同时补齐了 HotKnots 对 `_` 未配对符号的输入兼容。修复后上述 25,996 个数值全部与
C oracle 一致，最大绝对误差为 `1.98e-5 kcal/mol`；再跑 ArchiveII 的 78 个数值也
全部一致。

第四轮审计继续补齐此前只做过随机抽样的参数和拓扑空间，共新增 22,882 个合法
sequence/structure 输入、56,022 个模型 case、112,044 个
`energy`/`energy_no_dangling` 数值：完整枚举 1x2/2x1 内环的 4,608 种组合和 2x2
内环的 9,216 种组合（DP09 与 RE；CC09 的前 363 个普通热力学参数与 DP09 相同），
完整枚举 triloop/tetraloop、hairpin mismatch、外部 dangle 及单碱基 dangle 竞争，
并系统覆盖 GAIL、左右 bulge、长环外推和 multiloop dangle。拓扑部分另检查了 600 个
允许相邻 paired endpoint 的随机 crossing 图（五模型），以及长度 7--11 的全部
3,420 个合法 crossing matching（三模型）。112,044 个数值全部一致，breakdown 求和
不变量全部成立，最大绝对误差为 `2.1362e-5 kcal/mol`。最后再次重放 ArchiveII 的
13 个候选，78 个数值仍全部一致，最大绝对误差为 `2.16e-6 kcal/mol`。本轮没有发现
合法输入上的新公式、参数索引或 Loop/Bands 拓扑偏差。

CC 的 Python double 与原 C++ float 在未格式化内部值上最多约有 `2.4e-5 kcal/mol`
差异；原 `computeEnergy` 打印精度下结果相同。oracle 源码和二进制未保留在本分支，
其输出已固化为 `tests/test_hotknots_energy.py` 中的参考向量。

当前分支自身不含 `Utils/HotKnots_v2.0`；删除该目录后，26 个测试仍全部通过。
这证明的是“无 HotKnots 运行时依赖”，而不是“能量实现与 HotKnots 无来源关系”。
README 示例序列还通过了真实 Numba JIT 的两阶段端到端运行，并生成、计分和排序了
三个 pseudoknot 候选。日常验证命令为：

```bash
python3 -m pytest -q
python3 CPLfold.py -s GGCGCGGCACCGUCCGCGGAACAAACGG -b 20 -n1 3 -n2 2
```

## 10. 使用方式

```python
from Utils.hotknots_energy import HotKnotsEnergy

energy = HotKnotsEnergy().compute_energy(
    "GGCGCGGCACCGUCCGCGGAACAAACGG",
    "..(((((..[[[[)))))......]]]]",
    model="CC09",
)
print(energy["energy"])
print(energy["breakdown"])
```

`breakdown` 只包含可相加的 kcal/mol 能量项，其和等于 `energy`。诸如 CC 回退到 DP、
独立伪结组件数量等非能量信息位于 `metadata`，不会混入能量求和。

也可直接运行：

```bash
python -m Utils.hotknots_energy \
  -s GGCGCGGCACCGUCCGCGGAACAAACGG \
  --structure '..(((((..[[[[)))))......]]]]' \
  -m DP09
```

## 11. HotKnots 归属与许可说明

HotKnots 2.0 的 README 将原始实现归属于 Jihong Ren 和 Baharak
Rastegari，并说明 Cristina Pop 与 Mirela Andronescu 也进行了修改。
本 Python 能量计算器基于该项目的能量计算代码和参数，因此应保留
上游归属。相关 HotKnots 源码文件头声明了 GNU GPL version 2 or later；
再分发这个派生实现时应同时遵循适用的上游许可条款。

- HotKnots 2.0: https://www.cs.ubc.ca/labs/algorithms/Software/HotKnots/
- Ren, J., Rastegari, B., Condon, A., & Hoos, H. H. (2005). *HotKnots:
  Heuristic prediction of RNA secondary structures including pseudoknots*.
  RNA, 11(10), 1494–1504.
