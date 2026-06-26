# 2026-06-25-1

## 原始目标

1. Optimization.jl 做主训练rpnjl的 $T_0,a_1,a_2$ 参数
2. 训练集取 $s/T^3$,同时取HRG+LQCD数据
3.  划分温区，每个温区都分出 train/val/test，比例可以是6：2：2.
4.  损失函数包括 $s/T^3$, LQCD区域可以包含 $\sigma$,另一半loss是 $T_{p c}\thickapprox 160 MeV.$   

## 今日完成反馈

### 1. 严格训练框架

- 新增脚本：`rpnjl/train_poly_T0_strict.jl`
- 采用 `Optimization.jl` + `OptimizationOptimJL.NelderMead()` 做无导数优化。
- 训练参数为 polynomial Polyakov loop potential 中的：
  - $a_1$
  - $a_2$
  - $T_0$
- 数据集使用：
  - HRG: `fit_data/HRG.txt`
  - LQCD: `fit_data/hotqcd_1407_6387_table1_eos_origin_yerr.csv`
- 温区分层后再按 6:2:2 划分 train/val/test：
  - `hrg_low`
  - `lqcd_low`
  - `crossover`
  - `mid`
  - `high`
- loss 结构：
  - 主 loss: $s/T^3$ 的加权残差。
  - LQCD 使用表中给出的 $\sigma$。
  - HRG 使用人为 pseudo-error，避免低温区权重过强。
  - 额外约束：轻夸克凝聚给出的 $T_{pc}^{light}\approx160$ MeV。
  - 物理惩罚：负 $s/T^3$ 与明显越界的 Polyakov loop。

### 2. 当前最佳训练结果

最终采用的一组参数：

```text
a1 = -7.505272994
a2 = 0.0918489428288
T0 = 175.695929622 MeV
Tpc_light = 159.990286206 MeV
```

训练/验证/测试表现：

```text
train chi2/point = 1.13902320089
val   chi2/point = 1.02564940653
test  chi2/point = 1.04542405467
n_train = 58
n_val   = 13
n_test  = 11
retcode = MaxIters
```

说明：

- `retcode = MaxIters` 表示本次优化达到给定迭代上限，并非严格意义上的优化器收敛。
- 但 train/val/test 的 chi2/point 接近，说明当前分割下没有明显过拟合。
- $T_{pc}^{light}$ 已被约束在 160 MeV 附近。

输出文件：

```text
rpnjl/data/strict_poly_T0/dataset_split.csv
rpnjl/data/strict_poly_T0/polyT0_a1_m7p505273_a2_0p091849_T0_175p695930_summary.csv
rpnjl/data/strict_poly_T0/polyT0_a1_m7p505273_a2_0p091849_T0_175p695930_predictions.csv
```

### 3. EOS_muB0 输出结构

- 更新脚本：`rpnjl/EOS_muB0.jl`
- 使用给定的 $a_1,a_2,T_0$ 输出 $\mu_B=0$ 的 EOS。
- 输出 CSV 包含：
  - `P_over_T4`
  - `I_over_T4`
  - `s_over_T3`
  - `e_over_T4`
  - `nB_over_T3`
  - `CV_over_T3`
  - `cs2`
  - `phi_u, phi_d, phi_s`
  - `Phi, PhiBar`
  - solver residual
- 压强归一化采用：

```text
P(T=50 MeV, muB=0) = P_HRG(T=50 MeV)
```

- 同时输出 SVG，对比：
  - RPNJL 实线
  - HRG 虚线
  - LQCD 点和 error bar

输出文件：

```text
rpnjl/data/eos_muB0_poly/eos_muB0_poly_a1_m7p505273_a2_0p091849_T0_175p695930.csv
rpnjl/data/eos_muB0_poly/eos_muB0_poly_a1_m7p505273_a2_0p091849_T0_175p695930.svg
```

### 4. 接口同步

用户调整了 `rpnjl.jl` 的输入形式后，已同步如下接口：

- `Omega_param(orders, mu_B, T, ints, a1p, a2p)`
- `dOmega_dorder_param(orders, mu_B, T, ints, a1p, a2p)`
- `dOmega_dT_param(orders, mu_B, T, ints, a1p, a2p)`
- `Quark_mu_param(X0, mu_B, T, ints, a1p, a2p)`

内部统一使用：

```text
mu_u = mu_d = mu_s = mu_B / 3
```

同时把 `EOS_muB0.jl` 与 `train_poly_T0_strict.jl` 中的 `Omega_poly_T0` 也同步为 `mu_B` 标量输入，避免训练脚本、EOS 脚本和主模型接口不一致。

### 5. 验证记录

已完成的 smoke tests：

- `rpnjl.jl`
  - `Omega_param`
  - `dOmega_dT_param`
  - `Quark_mu_param`
- `EOS_muB0.jl`
  - `compute_eos`
  - $\mu_B=0$ 下 `nB_over_T3 = 0`
- `train_poly_T0_strict.jl`
  - `Omega_poly_T0`
  - `dOmega_dT_poly_T0`
- 完整运行：

```text
julia --project=. rpnjl\EOS_muB0.jl --p_num 80 --Tstep 2
```

运行成功并重新生成 CSV/SVG。Plots 给出 `No strict ticks found` 警告，仅与刻度选择有关，不影响输出文件。

### 6. 文件管理

`.gitignore` 已加入：

```text
.tmp/
muses_4D-TExS-v1.0.1/
rpnjl/data/strict_poly_T0/
```

说明：

- 严格训练输出目录暂时不进入 git。
- `rpnjl/data/eos_muB0_poly/` 目前未加入 ignore，因为它是本次 EOS 结果输出，可按需要决定是否纳入版本管理。

## 当前结构

```text
rpnjl/
  rpnjl.jl
    主 RPNJL 模型、Omega、gap equation、参数化 a1/a2 接口

  train_poly_T0_strict.jl
    严格训练入口
    负责 HRG+LQCD 数据、温区 train/val/test、Nelder-Mead 优化、Tpc 约束

  EOS_muB0.jl
    使用给定 a1/a2/T0 计算 muB=0 EOS
    输出 CSV 与 SVG

  data/
    strict_poly_T0/
      训练 split、summary、prediction 输出

    eos_muB0_poly/
      EOS CSV/SVG 输出
```

## 后续待办

1. 进一步确认 HRG pseudo-error 的设定是否物理合理，当前会显著影响低温区权重。
2. 对当前最优点继续提高 `maxiters` 或做多初值训练，确认 `retcode = MaxIters` 后是否还能下降。
3. 检查 EOS 中低温区由于压强归一化导致的 $I/T^4$、$e/T^4$ 行为，判断是否需要重新定义真空常数或分段归一化。
4. 对 $C_V/T^3$ 和 $c_s^2$ 做更细的数值稳定性检查，尤其是 crossover 附近。
5. 如果后续考虑 $T_0(\mu_B)$，建议先固定本次 $\mu_B=0$ 结果作为 baseline，再扩展到有限 $\mu_B$。
