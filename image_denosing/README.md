**基于吉布斯采样的图像降噪**

对于具有 $y\in[-1,1]^d$ 分布的二位图像，有条件概率分布 
$$p(y|b,w)=\frac{1}{Z}\prod_{i=1}^d\exp(b_iy_i)\prod_{j\in \text{nb}(i)}\exp(w_{ij}y_iy_j)$$

其中 $\text{nb(i)}$ 表示与像素 $i$ 相邻的像素， $b_i$ 为像素 $i$ 保持当前颜色权重， $w_{ij}$ 为像素 $i$ 颜色与像素 $j$ 颜色趋同权重。

从此前提可推出条件概率

$$p(y_i=1|y_{-i})=\frac{1}{1+\exp\left(-2b_i-2_{j\in\text{nb}(i)}w_{ij}y_j\right)}$$

以此满足吉布斯采样条件
