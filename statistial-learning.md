# chapter 1 统计学习方法概论
## 1.1 统计学习
#### 统计学习的定义
统计学习(statistical learning) 是关于 计算机基于数据 构建概率统计模型 并运用模型对数据进行预测和分析的一门学科。
#### 统计学习的主要特点
1. 统计学习以计算机及网络为平台，是建立在计算机及网络之上的；
2. 统计学习以 数据 为**研究对象**，是数据驱动的学科；
3. 统计学习的 **目的** 是对数据进行预测和分析；
4. 统计学习以 **方法** 为中心，统计学习方法构建模型并应用模型进行预测和分析；
5. 统计学习是概率论、统计学、信息论、计算理论、最优化理论及计算机科学等多个领域的交叉学科，并在发展中逐步形成独立的理论体系和方法论。

#### 统计学习的基本假设和前提
同类数据具有一定的统计规律性。

#### 统计学习的方法
+ 监督学习(supervised learning)
+ 非监督学习(unsupervised learning)
+ 半监督学习(semi-unsupervised learning)
+ 强化学习(reinforcement learning)

##### 实现统计学习的步骤
1. 得到一个有限的训练数据集合；
2. 确定包含所有可能的模型的假设空间，即学习模型的集合；
3. 确定模型选择的准则，即学习的策略；
4. 实现求解最优模型的算法，及学习的算法；
5. 通过学习选择最优模型；
6. 利用学习的最优模型对新数据进行预测或分析。

##### 统计学习3要素
+ 模型(model)
+ 策略(strategy)
+ 算法(algorithm)

#### 统计学习的研究
+ 统计学习方法(statistical learning method)
+ 统计学习理论(statistical learning theory)
+ 统计学习应用(application of statistical learning)

## 1.2 监督学习
统计学习 的任务是 学习一个模型，使模型能够对任意给定的输入，对其对应的输出做出一个好的预测。

### 1.2.1 基本概念

#### 1. 输入空间、特征空间、输出空间
将输入与输出所有可能取值的集合分别称为 输入空间(input space) 与 输出空间(output space).

每个具体的输入是 一个实例(instance),通常有特征向量(feature vector)表示.
所有特征向量存在的空间称为 特征空间(feature space).
特征空间的每一维对应一个 特征。

##### 预测任务分类
+ 回归问题 ：输入变量和输出变量均为连续变量的预测问题；
+ 分类问题 ：输出变量为有限个离散变量的预测问题；
+ 标注问题 ：输入变量和输出变量均为变量序列的预测问题。

#### 2. 联合概率分布
监督学习 假设输入和输出的随机变量X和Y遵循联合概率分布P(X,Y).
P(X,Y)表示分布函数，或分布密度函数。

#### 3. 假设空间
模型 属于有输入空间到输出空间的映射的集合，这个集合就是 假设空间(hypothesis space).
假设空间的确定 意味着 学习范围的确定。

监督学习的模型可以是 概率模型 或 非概率模型，
由条件概率分布P(Y|X)或决策函数(decision function)`Y=f(X)`表示.
对具体的输入进行相应的输出预测时，写作`P(y|x)`或`y=f(x)`.

### 1.2.2 问题的形式化
## 1.3 统计学习的三要素
方法 = 模型 + 策略 + 算法
### 1.3.1 模型
模型的假设空间(hypothesis space)包含所有可能的条件概率分布或决策函数。
### 1.3.2 策略
按照什么样的准则学习或选择最优的模型。
#### 1. 损失函数 和 风险函数
损失函数 度量模型一次预测的好坏；
风险函数 度量平均意义下模型预测的好坏。

1. 0-1损失函数(0-1 loss function)
2. 平方损失函数(quadratic loss function)
3. 绝对损失函数(absolute loss function)
4. 对数损失函数(logarithmic loss function) 或 对数似然损失函数(log-likelihood loss function)

#### 2. 经验风险最小化(empirical risk minimization, ERM) 与 结构风险最小化(structural risk minimization, SRM)
### 1.3.3 算法(algorithm)

## 1.4 模型评估 与 模型选择
### 1.4.1 训练误差(training error) 与 测试误差(test error)
泛化能力(generalization ability): 学习方法对未知数据的预测能力。
### 1.4.2 过拟合 与 模型选择(model selection)
#### 过拟合(over-fitting) :
如果一味追求提高对训练数据的预测能力，所选模型的复杂度则往往会比真模型更高。
过拟合 是指学习时选择的模型所包含的参数过多，以致于出现这一模型对已知数据预测得很好，但对位置数据预测得很差的现象。
模型选择旨在避免过拟合并提高模型的预测能力。

## 1.5 模型选择方法：正则化(regularization) 与 交叉验证(cross validation)
### 1.5.1正则化
正则化 是结构风险最小化策略的实现，是在经验风险上加一个正则化项(regularizer)或罚项(penalty term).

[奥卡姆剃刀(Occan's razor)原理](https://zh.wikipedia.org/wiki/%E5%A5%A5%E5%8D%A1%E5%A7%86%E5%89%83%E5%88%80)

### 1.5.2 交叉验证
随机将数据集切分为3部分：训练集(training set),验证集(validation),测试集(test set).

1. 简单交叉验证
2. S折交叉验证
3. 留一交叉验证

## 1.6 泛化能力
学习方法的泛化能力(generalization ability)是指由该方法学习到的模型对未知数据的预测能力，是学习方法本质上的重要性质。
### 1.6.1 泛化误差(generalization error)
### 1.6.2 泛化误差上界(generalization error bound)

## 1.7 生成模型 与 判别模型
生成模型

+ 朴素贝叶斯法
+ 隐马尔可夫模型

判别模型

+ k近邻法
+ 感知机
+ 决策树
+ 逻辑斯谛回归模型
+ 最大熵模型
+ 支持向量机
+ 提升方法
+ 条件随机场

## 1.8 分类问题
回归问题 ：输入变量和输出变量均为连续变量的预测问题；

## 1.9 标注问题
分类问题 ：输出变量为有限个离散变量的预测问题；

## 1.10 回归问题
标注问题 ：输入变量和输出变量均为变量序列的预测问题。

# chapter 2 感知机(perceptron)
感知机是二类分类的线性分类模型。
## 2.1 感知机模型
`f(x)=sign(w·x+b)`
w---权值(weight) 或 权值向量(weight vector)
b---偏置(bias)

## 2.2 感知机学习策略

### 2.2.1 数据集的线性可分性

## 2.3 感知学习算法

### 2.3.1 感知学习算法的原始形式

### 2.3.2 算法的收敛性

### 2.3.3 感知学习算法的对偶形式

# 2.4 扩展学习方法

+ 口袋算法(pocket algorithm)
+ 表决感知机(voted perceptron)
+ 带边缘感知机(perceptron with margin)

# chapter 3 k近邻法(k-nearest neighbor, k-NN)
是一种基本分类和回归方法。

给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的k个实例，这k个实例多数属于哪个类，就把该输入实例分为这个类。

三要素：

+ k值的选择
+ 距离量度
+ 分类决策规则

## 3.3 k近邻法的实现：kd树(kd tree)

### 3.3.1 构造kd树

### 3.3.2 搜索kd树

# chapter 4 朴素贝叶斯法(naive Bayes)

# chapter 5 决策树(decision tree)
分类 与 回归 方法

# chapter 6 逻辑斯谛回归(logistic regression) 与 最大熵模型(maximum entropy model)
对数线性模型

## 6.1 逻辑斯谛回归(logistic regression)

## 6.2 最大熵模型(maximum entropy model)

### 6.2.1 最大熵原理

# chapter 7 支持向量机(support vector machines, SVM)