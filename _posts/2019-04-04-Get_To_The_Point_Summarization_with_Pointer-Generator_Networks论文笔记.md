---
layout: post
title: "Get To The Point: Summarization with Pointer-Generator Networks论文笔记"
date: 2019-04-04
categories: NLP
tags: ["summarization"]
---



﻿### 论文：[Get To The Point: Summarization with Pointer-Generator Networks](http://arxiv.org/abs/1704.04368)解读



#### **Abstract类型的文本摘要存在的问题**：

+ 容易产生不准确的事实；
+ 倾向于重复相同的内容。

![](http://i2.bvimg.com/682738/36db882aa8eb3096.png)

*注：如上面所示，作为基线的seq2seq+Attention这样的纯粹的生成式模型存在1.无法生成OOV词汇；2.歪曲事实两个问题。Pointer Generator 存在重复内容的问题。在引入Coverage mechanism之后的Pointer-Gen+Coverage模型能够解决上面的三个问题。*

#### **论文的创新点**：

+ （1）使用一个混合的指针-生成器网络(hybrid pointer-generator network)，它可以通过指针从源文本复制单词，这有助于准确复制信息，同时保留通过生成器生成新单词的能力。
+ （2）使用覆盖率（coverage）来追踪摘要的内容，这有助于缓解重复的现象。



#### Introduction

​	文本摘要使用抽取和生成两种方法：抽取式和生成式。抽取式是从文章中选择关键的词句拼接而成，这种方式能够保证生成句子语法和准确性的下限。然而高质量的摘要往往需要分段、泛化或者结合知识，这种类型的摘要只能使用生成式方法。seq2seq模型有三个问题：不准确的再现事实，无法处理词汇表外的（OOV）的单词，重复自己。论文提出的模型可以在多句子的文本摘要中解决这三个问题。混合指针-生成器网络便于通过指向从源文本复制单词，这提高了OOV单词的准确性和处理能力，同时保留了生成新单词的能力。可以看作是提取和抽象方法之间的平衡。在消除重复方面，论文提出了覆盖向量（coverage vector）来跟踪和控制对源文本的覆盖。



#### seq2seq attention model

1.标准的seq2seq模型使用BiLSTM对源文本进行编码，然后使用Encoder hidden state和Decoder hidden state计算新的时间步的Attention分布，进而得到新的上下文向量context vector，使用softmax层对上下文变量解码得到新时间步的词汇分布。

+ 根据当前解码器的隐藏状态$s_t$和输入不同时间步的隐藏状态$h_i$分别计算得到attention分布$a^t​$

$$
e_i^t = v^T tanh(W_hh_i+W_ss_t+b_{att})
$$

$$
a^t = softmax(e^t)
$$

+ 计算attention分布对所有的$h_i$加权和

$$
h_t^* = \sum _i a_i^t h_i
$$

+ 使用$[s_t,h_t^*]​$计算最终的预测值

$$
P_{vocab} = softmax(V^{'}(V[s_t, h_t^*]+b)+b^{'})
$$

![在这里插入图片描述](http://i2.bvimg.com/682738/310feb2eea3b74b4.png)



#### Pointter-genearator network

1.pointer-generator网络的关键之处在于使用一个$p_{gen}$来表征从词汇表中生成当前时间步词汇的概率,$（1-p_{gen})$表示从源文本中拷贝词汇的概率。
2.$P_{vocab}(w)$表示使用标准seq2seq网络生成词汇的分布，$\sum_{i:w_i=w}a_i^t $表示上下文中所有当前词汇出现是其atttention的和。

+ 计算$P_{gen}​$

$$
P_{gen} = \sigma(w_{h^*}^Th_t^*+ w_s^Ts_t+ w_x^Tx_t+b_{ptr})
$$

+ 与seq2seq attention model相同，计算$P_{vocab}​$
+ 计算$P_{gen}$和$(1-P_{gen})$加权的$P(w)$

$$
P(w) = P_{gen}P_{vocab}(w)+(1-p_{gen}) \sum _{i:w_i =w}a_i^t
$$

![在这里插入图片描述](http://i2.bvimg.com/682738/379d25261c3c75ac.png)

*注：以上图为例，decoder已经输出了Germany beat两个词汇，这时候希望生成下一个时间步的词汇，如果目标词汇$w$未出现在原文中则$\sum _ {i:w_i=w} a_i^t=0$,* 如果目标词汇$w$不在词典中则$P_{vocab}(w)=0​$

#### Coverage mechanism

1.重复的问题在多句摘要中经常出现，论文中引入覆盖(coverage)机制来监控摘要中生成的词对源文本的覆盖情况，以减少重复关注一部分源文本进而生成重复内容的情况出现。

2.coverage vector $c_t= \sum_{t'=0}^{t-1} a^{t'}$等于历史time step的attention之和来表示decoder生成过的词汇的attention覆盖的情况。$c_t$作为下一个时间步计算attention的一个输入，所以pointer-gen网络计算attention那个步骤的公式变为:


$$
e_i^t = v^T tanh(W_hh_i+W_ss_t+ w_cc_i^t + b_{att})
$$


3.论文引入coverage损失对重复关注同一个位置进行的惩罚,公式为：


$$
covloss _t = \sum _i min(a_i^t, c_i^t)
$$


4.最终的loss是$P(w)$与covLoss之和:


$$
loss_t = -logP(w_t^*)+\lambda \sum _i min(a_i^t, c_i^t
$$

#### result
![在这里插入图片描述](http://i2.bvimg.com/682738/a33fd5a77bb46996.png)

