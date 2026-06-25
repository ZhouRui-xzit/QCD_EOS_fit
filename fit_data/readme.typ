#import "@preview/touying:0.6.1": *

#import "@preview/physica:0.9.5": *
#import themes.stargazer: *
#import "@preview/muchpdf:0.1.0": muchpdf


#import "@preview/numbly:0.1.0": numbly

#import "@preview/lilaq:0.5.0" as lq
#import "@preview/numty:0.0.5" as nt  // 数值计算包

#set text(font: ("Arial", "FZShuSong-Z01"), size: 14pt)

#show strong: text.with(font: ("Arial", "FZHei-B01"), size: 14pt)

#show emph: text.with(font: ("Arial", "FZKai-Z03"), size: 14pt)

#show math.equation: set text(font: "Latin Modern Math")

  #set heading(numbering: "1.1.")
  #set par(justify: true,first-line-indent: 2em) // 两端对齐，段前缩进2字符
  #show heading: it =>  {
  it
  par()[#text(size:0.5em)[#h(0.0em)]]
  }
  #show figure: it =>  {
      it
      par()[#text(size:0.5em)[#h(0.0em)]]
  }

    // 自定义二级标题样式
    #show heading.where(level: 1): it => {
      text(
        size: 20pt,               // 字体大小
        fill: blue,               // 字体颜色
        it
      )
       v(-10pt)
    }
    // 自定义二级标题样式
    #show heading.where(level: 2): it => {
      text(
        size: 16pt,               // 字体大小
        fill: blue,               // 字体颜色
        it
      )
       v(-10pt)
    }

    // 自定义三级标题样式
    #show heading.where(level: 3): it => {
      text(
        size: 14pt,               // 字体大小
        fill: blue,               // 字体颜色
        it
      )
      v(-10pt)
    }

  #set figure(numbering: (..nums) => {
    let ch = counter(heading).get().first()
    numbering("(1.1)", ch, ..nums)
  })


   #set math.equation(numbering: n => {
  
  // if you want change the number of number of displayed
  // section numbers, modify it this way:
  let count = counter(heading).get()
  let h1 = count.first()
  let h2 = count.at(1, default: 0)
  numbering("(1.1.1)", h1, h2, n)
})



    #set math.equation(supplement: [eq])
    #show figure.where(
    kind: table
    ): set figure(supplement: [table])
    #show figure.where(
    kind: table
    ): set figure.caption(position: top)
    #show figure.where(
    kind: image
    ): set figure(supplement: [fig])
    #show heading.where(level: 2): it => it + counter(math.equation).update(0)
  

    //show link: text.with(fill: mycolors.refcolor) // 网址链接
    #show ref: it =>{text(it,red,font: ("Libertinus Serif", "FZHei-B01"), size: 14pt)}

    #show ref: it => {
    if query(it.target).len() == 0 {
    return text(fill: red, "<???" + ">")
    }
    it
    }

    //ref 

    #set math.mat(row-gap:1em, column-gap:1em)

    // Page
    #set page(paper: "a4",fill: white, margin: (x:40pt, y:60pt), numbering: "1", header: context {
      v(-10pt)
      line(length: 100%, stroke: black) 
    }, footer: [
        #align(center,context {counter(page).display("1")})
        #line(length: 100%, stroke: black) 
      ])
    // page 
 // 修改脚注样式（在 mybook 函数中）
  #show footnote.entry: it => {
    set text(fill: mycolors.refcolor, size: 10pt)
    set math.equation(numbering: none)  // 脚注中的公式不编号
    show math.equation: set text(size: 10pt, font: "Libertinus Math")
    it
  }

= EOM of EMD
== 作用量和度规拟设
标准的Einstein-Maxwell-Dilaton作用量写作 $ 
    S = 1/(2 kappa_N^2) integral dd(x,5) 
    sqrt(-g) [R - 1/2 nabla_mu phi nabla^mu phi - Z(phi)/4 F_(mu nu) F^(mu nu) - V(phi)] 
  $<eq:EMD_action>
我们将bulk的度规写作  $ 
    dd(s^2) = (L^2 e^(2 A_E (z)))/z^2 [
      -f(z) dd(t,2) + 1/f(z) dd(z,2) + dd(vb(x),2)   
    ]
  $<eq:EMD_metric>
其中 $L$ 是AdS空间的半径, 不妨取作 $L=1$. 在@eq:EMD_metric 拟设下，黑洞视界 $z=z_h$ 由  $ 
    f(z_h) =0 
  $
确定，黑洞温度和黑洞熵密度为  $ 
    T=  abs(f'(z_h))/(4 pi), s = (2 pi e^(3 A_E (z_h)))/(kappa_N^2  z_h^3)
  $
下面分别对度规，标量场和电磁场变分可以得到运动方程。

首先我们注意到物质场  $ 
    cal(L)_"M" = -1/2 nabla_mu phi nabla^mu phi - Z(phi)/4 F_(mu nu) F^(mu nu) - V(phi) 
  $
由此物质场(裸)能动张量为  $ 
    T^(mu nu)= -2/sqrt(-g) pdv((sqrt(-g) cal(L)_"M" ), g_(mu nu),d:delta) &= 
      partial^mu phi partial^nu phi - 1/2 g^(mu nu) partial_rho phi partial^rho phi 
      - g^(mu nu) V(phi) \ 
      &+Z(phi) [ F^(mu rho) tensor(F,+nu,-rho)-1/4 g^(mu nu) F_(rho sigma) F^(rho sigma) ]
  $
由此，度规对应的运动方程，即Einstein场方程写作  $ 
    G_(mu nu) = 1/2 T_(mu nu) 
  $
即有   $ 
    R_(mu nu)-1/2 R g_(mu nu) &=  1/2 partial_mu phi partial_nu phi 
    +Z/2 F_(mu rho) tensor(F,-nu,+rho) \ 
    &+1/2 (-1/2 nabla_mu phi nabla^mu phi - V(phi) - Z/4 F_(rho sigma) F^(rho sigma)) g_(mu nu)
  $
此外，关于标量场和电磁场的运动方程分别为  $ 
     nabla_mu  nabla^mu phi - (partial_phi Z)/4 F_(mu nu) F^(mu nu) - partial_phi V = 0\ 
     nabla_mu (Z F^(mu nu)) = 0
  $<eq:EMD_EOM2>
我们假设电磁场只有时间分量，即 $A = A_a dd(t^a) $, 进一步所有物质场只在 $z$ 方向变换，即  $ 
    A_t equiv A_t (z), phi equiv phi (z) 
  $
由此电磁场张量只有 $F_(z t)=A'_t (z)$ 是非零的. 下面我们引入辅助函数  $ 
    B(z) = A_E (z) - ln z 
  $
则度规  $ 
    dd(s)^2 = e^(2 B(z)) [-f(z) dd(t,2) + 1/f(z) dd(z,2) + dd(vb(x),2)]  
  $
于是  $ 
    F^2 =F_(mu nu) F^(mu nu )= -2 e^(-4 B(z)) A'^2_t 
  $
注意到标量场的Laplace结果为  $ 
    nabla^2 phi &= 1/sqrt(-g) partial_z [sqrt(-g) g^(z z) partial_z phi] =
    e^(-5 B) partial_z [
      e^(3 B) f  phi'
    ]\ 
    &=e^(-2 B) f phi'' + e^(-2 B) (3 B' f + f') phi'
  $
代入@eq:EMD_EOM2 中的标量场方程，我们可以得到  $ 
    phi'' + (3 B' + f'/f) phi' + e^(-2 B)/f (partial_phi Z)/2 A'^2_t -e^(2 B)/f partial_phi V = 0
  $
对于电磁场部分，我们有  $ 
      nabla_mu (Z F^(mu nu))  &= 1/sqrt(-g) partial_mu [sqrt(-g) Z F^(mu nu)] \
     & = e^(-5 B) partial_z ( e^(5 B)Z  e^(-4 B) partial_z A_t)\
      &= e^(-5 B)  partial_z (e^B Z  partial_z A_t) = 0
  $
即  $ 
    A_t '' + [B'+(partial_phi Z(phi))/Z(phi) phi'] A_t ' = 0 
  $

下面计算Einstein方程。在 $B$ 记号下，非零的Einstein张量混合分量为  $
    tensor(G,+t,-t) &= e^(-2B) [
      3/2 B' f' + 3 f B'' + 3 f B'^2
    ]\
    tensor(G,+z,-z) &= e^(-2B) [
      3/2 B' f' + 6 f B'^2
    ]\
    tensor(G,+i,-i) &= e^(-2B) [
      1/2 f'' + 3 B' f' + 3 f B'' + 3 f B'^2
    ] comma quad i=1,2,3 .
  $
另一方面，物质能动张量对应的混合分量为  $
    tensor(T,+t,-t) &= -1/2 e^(-2B) f phi'^2 - V(phi)
      - 1/2 Z(phi) e^(-4B) A_t'^2 \
    tensor(T,+z,-z) &= 1/2 e^(-2B) f phi'^2 - V(phi)
      - 1/2 Z(phi) e^(-4B) A_t'^2 \
    tensor(T,+i,-i) &= -1/2 e^(-2B) f phi'^2 - V(phi)
      + 1/2 Z(phi) e^(-4B) A_t'^2 .
  $
因此，由 $tensor(G,+z,-z)-tensor(G,+t,-t)=1/2(
tensor(T,+z,-z)-tensor(T,+t,-t))$ 可得  $
    B'' - B'^2 + 1/6 phi'^2 = 0 .
  $
由 $tensor(G,+i,-i)-tensor(G,+t,-t)=1/2(
tensor(T,+i,-i)-tensor(T,+t,-t))$ 可得  $
    f'' + 3 B' f' - e^(-2B) Z(phi) A_t'^2 = 0 .
  $

最后将辅助函数 $B(z)= A_E (z)-ln z$ 完全代回去，得到Fang度规下标准EMD系统的四组运动方程  $
    & f'' + (3 A_E' - 3/z) f'
      - z^2 e^(-2 A_E) Z(phi) A_t'^2 = 0,\
    & A_E'' - A_E'^2 + (2 A_E')/z + 1/6 phi'^2 = 0,\
    & A_t'' + (A_E' - 1/z + (partial_phi Z(phi))/Z(phi) phi') A_t' = 0,\
    & phi'' + (f'/f + 3 A_E' - 3/z) phi'
      + z^2 e^(-2 A_E)/(2 f) partial_phi Z(phi) A_t'^2
      - e^(2 A_E)/(z^2 f) partial_phi V(phi) = 0 .
  $<eq:fang_metric_emd_eom>

其中 $Z(phi)$ 和 $V(phi)$ 仍然取Cai模型中的规范化。作为检查，剩余的Einstein分量可以写成一条约束方程  $
    &3 (A_E' - 1/z) f'
    + 12 f (A_E' - 1/z)^2
    - 1/2 f phi'^2\
    & + e^(2 A_E)/z^2 V(phi)
    + 1/2 z^2 e^(-2 A_E) Z(phi) A_t'^2 = 0 .
  $

== UV渐近展开
取
$
  V(phi) &=-12 cosh(c_1 phi)+(6 c_1^2-3/2) phi^2+c_2 phi^6\ 

  Z(phi) &=1/(1+c_3) sech(c_4 phi^3)
  +c_3/(1+c_3) e^(-c_5 phi).
$
将
$
  f(z)&=1-f_4 z^4+f_6 z^6+dots,\
  A_E (z)&=A_2 z^2+(A_4+A_(4L) ln z)z^4+dots,\
  A_t (z)&=mu_B+q_2 z^2+q_3 z^3+q_4 z^4+dots,\
  phi (z)&=p_1 z+(p_3+p_(3L) ln z)z^3+dots
$
代入@eq:fang_metric_emd_eom，可以得到
$
  A_2 &= -p_1^2/36,\
  p_(3L) &= (1-6 c_1^4) p_1^3/6,\
  A_(4L) &= -(1-6 c_1^4) p_1^4/120,\
  A_4 &= -p_1 p_3/20+(73-378 c_1^4)p_1^4/64800,\
  q_3 &= (2 c_3 c_5 p_1)/(3(1+c_3)) q_2,\
  q_4 &= p_1^2 [1/72+c_3(c_3-1)c_5^2/(4(1+c_3)^2)] q_2,\
  f_6 &= q_2^2/3-p_1^2 f_4/18 .
$
因此可以写作
$
  f(z) &= 1-f_4 z^4
    +(q_2^2/3-p_1^2 f_4/18)z^6+dots,\
  A_E (z) &= -p_1^2/36 z^2
    +[-p_1 p_3/20+(73-378 c_1^4)p_1^4/64800]z^4
    -(1-6 c_1^4)p_1^4/120 z^4 ln z+dots,\
  A_t (z) &= mu_B+q_2 z^2
    +(2 c_3 c_5 p_1)/(3(1+c_3))q_2 z^3
    +p_1^2 [1/72+c_3(c_3-1)c_5^2/(4(1+c_3)^2)]q_2 z^4+dots,\
  phi (z) &= p_1 z+p_3 z^3
    +(1-6 c_1^4)p_1^3/6 z^3 ln z+dots .
$
上述系数用 `para_fit/cai_rebuild_uv_neutral.wl` 和
`para_fit/cai_rebuild_uv_expansion.wl` 重新做过 Mathematica 逐阶检查。
检查方式是把每条运动方程拆成加法项，先逐项做 Laurent--log 展开，
再抽取 $z^n (ln z)^k$ 的系数；如果直接对完整残差整体调用 `Series`，
Mathematica 会在含 $1\/z$、$ln z$ 和指数因子的表达式中漏掉低阶贡献。
中性情形得到的差异列表为
$
  {0,0,0,0,0,0,0,0},
$
带电情形得到的差异列表为
$
  {0,0,0,0,0,0,0,0,0,0}.
$
因此 $mu_B=0$ 的展开与当前 rebuild 代码使用的中性系数一致；
推广到 $mu_B != 0$ 后，$q_3$ 与原 `cai_2022_spectral` 中
$A_t$ 的 UV 重定义因子一致，而 $q_4$ 这里应使用 Fang gauge
方程直接给出的 $p_1^2/72$ 常数项；旧的 $eta,F$ 变量写法中该常数项
会被度规变量重定义改写。这里 $c_4$ 不出现在 $q_3,q_4,f_6$
这些低阶系数中，因为 $sech(c_4 phi^3)=1+O(phi^6)$。
按照现有Cai归一化，$q_2=-kappa_N^2 n_B$。

== Fang gauge下的全息重整化
为了避免在不同径向规范之间来回映射，下面直接在
@eq:EMD_metric 的 Fang gauge 中计算边界能动张量。取 cutoff 面
$z=epsilon$，诱导度规为
$
  h_(t t)=- e^(2 A_E)/z^2 f,
  quad
  h_(i j)= e^(2 A_E)/z^2 delta_(i j).
$
由于边界位于 $z -> 0$，体区域为 $z >= epsilon$，外法向量取
$
  n^z=-z sqrt(f) e^(-A_E).
$
于是外曲率为
$
  K_(a b)=1/2 cal(L)_n h_(a b)
  =-z sqrt(f) e^(-A_E)/2 partial_z h_(a b).
$
在均匀化学势背景 $A=A_t(z) dif t$ 下，边界诱导场强的
Maxwell 对数 counterterm 不贡献到 $T_(t t)$ 和 $T_(i i)$。
采用 Cai 模型中的 counterterm 后，重整化 Brown--York 张量可写作
$
   T_(a b) 
  =1/(2 kappa_N^2) lim_(z->0) z^(-2) [
      & 2(K h_(a b)-K_(a b)-3 h_(a b))
      \
      &-(1/2 phi^2
      -(6 c_1^4-1)/12 phi^4 ln z
      +b phi^4) h_(a b)
    ] .
$
将上一节 UV 展开代入并转到 FG cutoff 后，若所有系数都按
$ln z$ 展开定义，有限项为
$
  (2 kappa_N^2) epsilon
  &= 30 A_4+3 f_4
    +(-133/2160+b+3 c_1^4/10)p_1^4
    +p_1 p_3,\ 
  (2 kappa_N^2) P
  &= -30 A_4+f_4
    +(133/2160-b-3 c_1^4/10)p_1^4
    -p_1 p_3 .
$
这里的 $p_3,A_4$ 是 $ln z$ 方案下的有限系数。当前
`EMD_cai_rebuild` 的数值重定义使用 $ln(z Lambda)$，因此从数值
边界值读出的有限系数还必须回到 Cai 的 $ln z$ 方案。另一个
数值上更敏感的点是：Fang gauge 中 $A_4$ 不是独立的物理 vev，
而是由 UV 方程约束给出的冗余系数。实际 renormalized pressure
应使用
$
  A_4^"UV"=-p_1 p_3/20+(73-378 c_1^4)p_1^4/64800,
$
而不是直接读取谱解边界的 $A_4^"raw"$。后者在转折区会把很小的
UV 约束误差通过 $f_v$ 中的 $-10 A_4$ 放大成明显的 $P$ 偏差。
令
$
  alpha=1-6 c_1^4,\ 
  p_(3,L)=alpha p_1^3/6,\ 
  A_(4,L)=-alpha p_1^4/120,\ 
  F_(4,L)=alpha p_1^4/12 .
$
Fang gauge 的直接重整化可等价写成先构造
$
  phi_v^F
  &= p_3 - p_1^3/36 + p_(3,L) ln Lambda,\ 
  f_v^F
  &= -f_4+2 A_2^2-10 A_4^"UV"-2 A_(4,L)
     +F_(4,L) ln Lambda,
$
其中 $A_2=-p_1^2/36$。于是
$
  P_F
  &=1/(2 kappa_N^2) [
    -f_v^F+p_1 phi_v^F
    +(3-48 b-8 c_1^4)p_1^4/48
  ],\
  epsilon_F
  &=1/(2 kappa_N^2) [
    -3 f_v^F+p_1 phi_v^F
    +(1+48 b)p_1^4/48
  ],\
  I_F&=epsilon_F-3 P_F .
$
这不是额外引入真空压强，而是把 $ln(z Lambda)$ 数值方案转换回
Cai 全息重整化公式使用的 $ln z$ 有限系数。修正后，$mu_B=0$
扫描中 $P_F/T^4$ 与旧 Cai gauge 的重整化压力在 $T=400,200,100$
MeV 处分别一致到约 $10^(-4),10^(-4),10^(-2)$ 的量级；剩余差异
主要来自低温积分零点和离散步长。
