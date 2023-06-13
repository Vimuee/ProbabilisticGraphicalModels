**变分自动编码器(VAE)**

此项目使用变分自动编码器识别并生成手写体的0，1数字。与自动编码器（AE）不同，VAE可以学习训练数据集的分布并从分布中采样新数据。此模型采用置信度下限


$$ELBO\left(x^{(i)}\right)=\mathbb{E}_{z^{(i)}~q\left(z^{(i)}|x^{(i)}\right)}\left[\log p\left(z^{(i)},x^{(i)}\right)-\log q\left(z^{(i)}|x^{(i)}\right)\right]$$

$$ELBO\left(x^{(i)}\right)=\mathbb{E}_{z^{(i)}~q\left(z^{(i)}|x^{(i)}\right)}\left[\log \mathcal{N}\left(z^{(i)};0,I\right)\mathcal{Bern}\left(x;f(z)\right)-\log \mathcal{N}\left(z;g(x), 0.01I\right)\right]$$

$f, g$ 是编码/解码器

$$f(z)=\sigma(w_2\text{ReLU}(w_1z+b_1)+b_2)$$

$$g(x)=w_4\text{ReLU}(w_3x+b_3)+b_4$$

隐变量空间维度为784

从jupyter notebook里的图像分析可看到，VAE成功将不同标签的数字分开，并且可以生成清晰的数字图片
