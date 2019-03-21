---
layout: post
title:  "《Pretraining-Based Natural Language Generation for Text Summarization》论文解读"
date: 2019-03-21 21:35:10 +0800
categories: notes
tags: paper
author: Henryzhou
---



### 《Pretraining-Based Natural Language Generation for Text Summarization》

国防科技大学 & MSRA	2019-2-26

#### 问题定义

摘要是指： “一段从一份或多份文本中提取出来的文字，它包含了原文本中的重要信息，其长度不超过或远少于原文本的一半”。自动文本摘要旨在通过机器自动输出简洁、流畅、保留关键信息的摘要。

自动文本摘要通常可分为两类，分别是**抽取式（extractive）**和**生成式（abstractive）**。抽取式摘要判断原文本中重要的句子，抽取这些句子成为一篇摘要。而生成式方法则应用先进的自然语言处理的算法，通过转述、同义替换、句子缩写等技术，**生成更凝练简洁的摘要**。比起抽取式，生成式更接近人进行摘要的过程。历史上，抽取式的效果通常优于生成式。伴随深度神经网络的兴起和研究，基于神经网络的生成式文本摘要得到快速发展，并取得了不错的成绩。

#### 生成式文本摘要的问题

**问题一**：在监督式训练中，对一篇文本一般往往只提供一个参考摘要，基于 MLE 的监督式训练只鼓励模型生成一模一样的摘要，然而正如在介绍中提到的，对于一篇文本，往往可以有不同的摘要，因此监督式学习的要求太过绝对。

针对这个问题论文作者提出了两个解决问题：1.两阶段的summary生成方法；2.使用mixed objective优化误差。



#### 论文创新点

**创新点1：两阶段的summary生成方法**

提出一种新颖的基于预训练模型的seq2seq的encoder-decoder框架，具体来说是基于BERT的两阶段产生文本摘要的方法。在编码器部分使用BERT对输入编码得到context representation；解码器部分分为两个阶段，首先使用Transformer-based的decoder产生summary的草稿，第二阶段分别将各个词汇做mask然后拼接上原input对mask部分的单词做单独的预测，这个阶段可以看做是一个fine-tune的过程。实验结果表示这种新型的模型在CNN/Daily和New York Time两个主流的数据集上面获得了SOAT的成绩。

**创新点2：mixed objective**

模型的另一创新，是提出了混合式学习目标，融合了监督式学习（teacher forcing）和强化学习（reinforcement learning）。考虑生成文本的流畅或者自然性，我们应该去优化生成文本的最大似然概率$ P(a|X) $，最大化句子中单词的联合概率分布，从而使模型学习到语言的概率分布。对于文本摘要我们还要评估一篇摘要的质量。对于一篇摘要而言，很难说有标准答案。不同于很多拥有客观评判标准的任务，摘要的评判一定程度上依赖主观判断。因此除了语法正确性、语言流畅性、关键信息完整度等标准，摘要的评价还需要定义一些新的文本摘要评估的指标，论文中使用的ROUGE评估指标，它会计算一个Reward值来表征summary文本和ground true的相关性。

#### ROUGE评估方式

评价文本摘要的质量的标准目前主要有以下三条：

1. 决定原始文本最重要的、需要保留的部分；
2. 在自动文本摘要中识别出 1 中的部分；
3. 基于语法和连贯性（coherence）评价摘要的可读性（readability）。

为了更高效地评估自动文本摘要，可以选定一个或若干指标（metrics），基于这些指标比较生成的摘要和参考摘要（人工撰写，被认为是正确的摘要）进行自动评价。目前最常用、也最受到认可的指标是 ROUGE（Recall-Oriented Understudy for Gisting Evaluation）。ROUGE 是 Lin 提出的一个指标集合，包括一些衍生的指标，最常用的有 ROUGE-n，ROUGE-L，ROUGE-SU：

- ROUGE-n：该指标旨在通过比较生成的摘要和参考摘要的 n-grams（连续的 n 个词）评价摘要的质量。常用的有 ROUGE-1，ROUGE-2，ROUGE-3。

- ROUGE-L：不同于 ROUGE-n，该指标基于最长公共子序列（LCS）评价摘要。如果生成的摘要和参考摘要的 LCS 越长，那么认为生成的摘要质量越高。该指标的不足之处在于，它要求 n-grams 一定是连续的。

- ROUGE-SU：该指标综合考虑 uni-grams（n = 1）和 bi-grams（n = 2），允许 bi-grams 的第一个字和第二个字之间插入其他词，因此比 ROUGE-L 更灵活。

  如果一篇生成的摘要恰好是在参考摘要的基础上进行同义词替换，改写成字词完全不同的摘要，虽然这仍是一篇质量较高的摘要，但 ROUGE 值会呈现相反的结论。为了避免上述情况的发生，在 evaluation 时，通常会使用几篇摘要作为参考和基准，这有效地增加了 ROUGE 的可信度，也考虑到了摘要的不唯一性。



#### 模型：

![](https://inews.gtimg.com/newsapp_bt/0/7925785315/1000.jpg)

输入的document input：$ X=\{x_1, ..., x_m\} $， 对应的summary表示为$ Y=\{y_1, ..., y_L\}​$ ， 

**Encoder**

encoder使用的是BERT(base)，输入X的enbedding为$ H= \{ h_1, ..., h_m \} = BERT(x_1, ..., x_m)$ 

**第一阶段：Summary draft decoder**

使用一个单独的left-to-right的decoder生成summary draft，这个decoder是一个N层Transformer的网络，具体输入为在时间t是将已经生成的$ summary：\{y_1, ..., y_{t-1}\}$ 通过BERT转化成$ embedding：\{q_1, ... q_{t-1}\}$ 以及document embedding，经过T个时间步能够生成长度为L的文本摘要，然后在[PAD]处截断，损失函数为
$$
P_t^{vocab} (w) = f_{dec}(q_{<t}, H) \ \ \ \ \  \ \ \ \ \ \ L_{dec}=\sum _{i=1} ^{|a|} -log P(a_i = y_i ^*|a_{<t}, H)
$$
**引入mixed objective：**

前面说明过只是用极大似然损失函数优化网络会使得模型过于死板，用于评价生成摘要的 ROUGE 指标却能考虑到MLE损失函数灵活性不足的问题，通过比较参考摘要和生成的摘要，给出摘要的评价（见下文评估摘要部分）。模型在训练时引入 ROUGE 指标。但由于 ROUGE 并不可导的，传统的求梯度 + backpropagation 并不能直接应用到 ROUGE。因此，作者利用强化学习将 ROUGE 指标加入训练目标。
$$
L_{dec}^{rl} = R(a^s).[-log(P(a^s|x)]
$$

$$
\hat{L}_{dec} = \gamma *L_{dec}^{rl} + (1-\gamma)*L_{dec}
$$



最终的训练目标是最大似然和基于 ROUGE 指标的函数的加权平均，这两个子目标各司其职：最大似然承担了建立好的语言模型的责任，使模型生成语法正确、文字流畅的文本；而 ROUGE 指标则降低 exposure bias，允许摘要拥有更多的灵活性，同时针对 ROUGE 的优化也直接提升了模型的 ROUGE 评分。

**第二阶段：refine draft：**

refine阶段的目标是使用BERT对masked LM任务的天然适配特性来进一步的优化summary的质量，这个阶段依然使用BERT生成embedding，与第一阶段不同的是做masked之后的summary draft。每一个时间步refine一个masked位置的词汇，n个时间步完成refine过程。refine阶段的损失函数为：
$$
L_{refine} = \sum_{i=1}^{|y|}-log P (y_i=y^*|a_{\ne i}, H)
$$
其中$ y_i $表示refine阶段第t时间点预测的masked的词汇，$ y^*$表示ground true的单词。$ a_{\ne i} = {a_1, ..., a_{i-1}, a_{i+1},..., a_{|y|}}$即表示出去masked单词之外的summary draft。这个过程可以描述为通过document生成summary draft，然后在summary draft基础上做refine summary。这个refine的阶段灵感来源于BERT的masked LM预训练任务，使用BERT的预训练网络的权重就能有效的利用到大量的无监督文本的上下文信息使得摘要更加流畅自然。

#### 其他小技巧

1. 共享解码器权重，加快训练时模型的收敛；
2. 人工规则，规定不能重复出现连续的三个词。




#### 参考文献：

[当深度学习遇见自动文本摘要](http://www.raincent.com/content-10-9344-3.html)

[Pretraining-Based Natural Language Generation for Text Summarization](https://arxiv.org/pdf/1902.09243.pdf)