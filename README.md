# Paper Summary  

- ## (*ECCV2018_SCAN*) Stacked Cross Attention for Image-Text Matching. [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)] [[code](https://github.com/kuanghuei/SCAN)]  
    - ### 思路  
        首先对图片中的区域(region)和句子中的单词(word)进行编码并将它们映射到相同的嵌入空间(embedding space)，然后通过Stacked Cross Attention 进行regions和words的对齐，求出图片和句子的相似度。  
    - ### 创新点  
        先前的工作大多是先求出图片中的每个region和句子中的每个word的相似度，然后将结果聚合起来作为图片和句子的相似度，但不同的区域或单词的重要程度是不一样的。本文最大的创新/贡献就是引入了Stacked Cross Attention机制，使用regions和words互为上下文进行attention操作，来考虑不同区域或单词的重要程度。  
    - ### Stacked Cross Attention  
        本文给出了两种SCA范式：Image-Text SCA 和 Text-Image SCA，此处主要介绍 Image-Text SCA.  
        SCAN有两个输入：图片特征V={v<sub>1</sub>, ..., v<sub>k</sub>}和单词特征E={e<sub>1</sub>, ..., e<sub>n</sub>}，其中v<sub>i</sub>表示图片第i个region的特征，e<sub>j</sub>表示句子第j个单词的特征，v<sub>i</sub>和e<sub>j</sub>维度相同。  
        - #### Image-Text Stacked Cross Attention  
            ![](./images/Cross-Modal-Interaction/SCAN/1.png)  
            - ##### 第一阶段：以图像区域作为上下文来attend句子中的单词  
                计算每一个region和每一个word的余弦相似度：  
                ![](./images/Cross-Modal-Interaction/SCAN/2.png)  
                对相似度矩阵进行归一化处理：  
                ![](./images/Cross-Modal-Interaction/SCAN/3.png)  
                计算对于图片中的第i个区域，句子中的第j个单词在整个句子中的重要程度，即注意力分数：  
                ![](./images/Cross-Modal-Interaction/SCAN/5.png) 注：λ<sub>1</sub>为softmax函数[温度参数](https://zhuanlan.zhihu.com/p/132785733)的倒数。  
                对句子中的所有单词加权求和得到句子向量，得到的句子向量就考虑了每个单词的重要程度：  
                ![](./images/Cross-Modal-Interaction/SCAN/4.png)  
            - ##### 第二阶段：以句子作为上下文来attend图片中的区域  
                计算图片中的每个区域和其对应的句子向量之间的相似度：  
                ![](./images/Cross-Modal-Interaction/SCAN/6.png)  
            - ##### 计算整张图片和整个句子的相似度  
                文中给出了LogSumExp pooling(LSE)和average pooling(AVG)两种方式计算图片和句子的相似度，分别为：  
                ![](./images/Cross-Modal-Interaction/SCAN/7.png)  
                ![](./images/Cross-Modal-Interaction/SCAN/8.png)  
                注：LSE中的λ<sub>2</sub>用来提高 R大的项的重要性，当 λ<sub>2</sub> → 无穷大时，S(I, T)就近似等于最大的R.  
        - #### Text-Image Stacked Cross Attention  
            ![](./images/Cross-Modal-Interaction/SCAN/12.png)  
            与第一种范式类似，不再赘述。  
    - ### Alignment Objective  
        使用 triplet loss 作为总的损失函数来训练模型：  
        ![triplet-loss](./images/Cross-Modal-Interaction/SCAN/9.png)  
        负样本的选取方式如下：  
        ![](./images/Cross-Modal-Interaction/SCAN/10.png) ![](./images/Cross-Modal-Interaction/SCAN/11.png)  
        即选取除了本身anchor之外相似度最高的图文作为负样本。  

