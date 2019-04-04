---
layout: post
title: "《The Evolved Transformer》论文总结"
date: 2019-03-28
categories: NLP
tags: ["Feature_Extractor"]
---



### Google Brain-[The Evolved Transformer](http://arxiv.org/abs/1901.11117) 

#### 论文创新点：

​	使用神经架构搜索的方法，为 seq2seq 任务找到了一种比Transformer更好的前馈网络架构。架构搜索是基于Transformer进行演进，最终得到的Evolved Transformer 的新架构在四个成熟的语言任务（WMT 2014 英德、WMT 2014 英法、WMT 2014 英捷及十亿词语言模型基准（LM1B））上的表现均优于原版 Transformer。在用大型模型进行的实验中，Evolved Transformer 的效率（FLOPS）是 Transformer 的两倍，而且质量没有损失。在更适合移动设备的小型模型（参数量为 7M）中，Evolved Transformer 的 BLEU 值高出 Transformer 0.7。



#### 搜索空间：

​	一个模型包含encoder和decoder，各包含若干个单元，编码器的单元包含6个模块，解码器的单元包含8个模块。每个模块分左右两个分支，各自接受一个隐藏状态作为输入。按照层次从低到高分支搜索项分为：input、normalization、layer、output dimension和activation。左右分支通过combiner function合并为新的隐藏状态作为输出。

![在这里插入图片描述](http://i2.bvimg.com/682738/433f27b8a361812e.png)
+ **Input**：分支可以从输入池中选择一个隐藏状态作为当前block的输入。单元中的第i个block可以从[0, i]个隐藏状态中进行选择，其中第j个隐藏状态表示该cell中第j个block的输出，第0个候选项为单元的输入。
+ **Normalization**：归一化项提供了两个选项， [LAYER NORMALIZATION (Ba et al., 2016), NONE]
+ **Layer**：构造一个神经网络层，提供的选项包括：
  + 标准卷积
  + 深度可分离卷积
  +  LIGHTWEIGHT 卷积
  + n头注意力层
  + GATED LINEAR UNIT 
  + ATTEND TO ENCODER（decoder专用）
  + 全等无操作
  + Dead Branch，切断输出
+ **Relative Output Dimension**：决定神经网络层输出的维度。
+ **Activation**：搜索中激活函数的选项有[SWISH, RELU, LEAKY RELU,  NON]
+ **Combiner Function**：表征的是左枝和右枝的结合方式，包括{ADDITION、CONCATENATION、MULTIPLICATION}。如果左右枝最终输出形状不同，则需要使用padding进行填充。短的向量向长的向量对齐，当使用加法进行结合时使用0填充，当使用乘法进行结合时使用1填充。
+ **Number of cells**：纵向叠加的cell的数量，搜索范围是[1,6]

#### 演进的过程：

+ 锦标赛选择（Tournament Selection）：

  + tournament selection算法是一种遗传算法，首先随机生成一批个体, 这些个体是一个个由不同组件组成的完整的模型，我们在目标任务上训练这些个体并在验证集上面计算他们的表现。
  + 首先在初始种群中进行采样产生子种群，从子种群中选出适应性（fitness）最高的个体作为亲本（parent）。被选中的亲本进行突变——也就是将网络模型中的一些组件改变为其他的组件——以产生子模型，然后在对这些子模型分配适应度（fitness），在训练集和测试集上进行训练和验证。
  + 对种群重新进行采样，用通过评估的子模型代替子种群中的fitness的个体以生成新的种群。
  + 重复上面的步骤，直到种群中出现超过给定指标的模型。

+ 渐进式动态障碍（Progressive Dynamic Hurdle）：

  ​	实验使用的训练集是WMT14英语到德语的机器翻译数据集，完整的训练和验证过程需要很长的时间，如果在所有的子模型上进行完整的训练和验证过程将会耗费很大的计算资源。因此论文中使用渐进式动态障碍的方法来提前停止一些没有前景的模型的训练，转而将更多的计算资源分配那些当前表现更好的子模型。具体来说就是让当前表现最好的一些模型多训练一些step。

  ​	假设当前种群经过一次锦标赛选择，生成了m个子模型并且加入到了种群中，这时候计算整个种群fitness的平均值$h_0$,下一次锦标赛选择将会以$h_0$作为对照，生成的另外m个fitness超过$h_0$的子模型可以继续训练$s_1$个step，接着进行种群中的所有的其他个体会继续训练$s_1$个step，然后在新的种群中生成$h_1$，以此类推知道种群中所有的个体的训练step都达到一个指定值。

  ​	如果一个子模型是由第$i$次锦标赛选择之后的亲本生成的，那么验证的过程将会进行$i$次。第一次为该模型分配$s_0$次的训练step并且在验证集上进行验证，若验证的fitness大于$h_0$则再分配$s_1$次训练step，再验证，再与$h_1$比较，只有子样本通过${h_0, h_1, ..., h_i}$次比较才能作为新的个体加入到新的种群中。

  

![在这里插入图片描述](http://i2.bvimg.com/682738/9ac3d366ada01c1a.png)



