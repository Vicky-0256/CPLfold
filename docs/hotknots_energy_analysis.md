# HotKnots 能量计算抽取与 Python 化分析

## 1. CPLfold 实际使用了什么

CPLfold 原先通过 `Utils/HotKnots_v2.0/hotknots.py` 启动
`bin/computeEnergy`。它只调用 `compute_energy()` 给一个已知结构打分，完全没有调用
HotKnots 的 hotspot 生成、启发式搜索、候选结构扩展或绘图代码。因此需要保留的是：

```text
CPLfold.py
  -> compute_energy(sequence, structure, model)
     -> dot-bracket 配对表和 loop/band 分解
     -> SimFold FM363 普通二级结构能量
     -> DP / CC / RE 伪结修正
     -> exterior dangling-end 修正
```

新的 `Utils/hotknots_energy.py` 从独立的 `Utils/energy_params` 读取所需参数子集并
完成上述计算，不再导入 HotKnots 包、读取其目录或启动原生可执行文件。仓库里原
`computeEnergy` 是 AArch64 ELF；Python 版也消除了运行机器与二进制架构必须一致的
问题。完成差分验证后，`Utils/HotKnots_v2.0` 已从独立实现分支删除。

## 2. 完整 closed-region / Loop / Bands 树

只按括号类型寻找两条 stem 只能覆盖简单 H 型。完整实现采用与原 `Stack`、`Loop`
和 `Bands` 相同的结构语义：

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

DP、CC 使用各自参数文件的前 363 项；RE 使用
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

中央环为 0 或 1 nt 时还会分别计算 flush 或 mismatch coaxial stacking。以下情况与
HotKnots 一样自动退回参数文件中携带的 DP 模型：stem 长度不在 2–12、需要的短环
表项缺失、loop1/loop2 为 0，或拓扑不适合 CC 表。多于两个 band、kissing/chain、
band span 中有 multiloop 等情况都走该明确定义的回退路径，并不是未计分或近似跳过。

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

输入仍须是长度一致的 RNA 序列和成对的 dot-bracket，且所有 pair 必须是 AU、CG 或
GU canonical pair。`UnsupportedTopologyError` 仅为旧版调用方保留；上述合法复杂拓扑
不再触发它，CPLfold 也不再以“unsupported”跳过候选。

## 9. 与原程序的验证

开发时在 x86-64 上重新编译了未改变算法的 HotKnots 2.0，并逐结构比较两列输出：

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

CC 的 Python double 与原 C++ float 在未格式化内部值上最多约有 `2.4e-5 kcal/mol`
差异；原 `computeEnergy` 打印精度下结果相同。固定参考向量保存在
`tests/test_hotknots_energy.py`。

运行时独立性也做了隔离验证：只复制 `CPLfold.py`、Python `Utils`、测试和
`Utils/energy_params` 到不含 `HotKnots_v2.0` 的临时目录，19 个测试全部通过。
README 示例序列还通过了真实 Numba JIT 的两阶段端到端运行，并生成、计分和排序了
三个 pseudoknot 候选。

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
