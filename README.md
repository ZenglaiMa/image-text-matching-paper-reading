# Paper Summary
---
- ### (*ECCV2018_SCAN*) Stacked Cross Attention for Image-Text Matching. [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)] [[code](https://github.com/kuanghuei/SCAN)]
    - #### 思路
        首先对图片中的区域(regions)和句子中的单词(words)进行编码并将它们映射到相同的嵌入空间(embedding space)，然后通过
        Stacked Cross Attention 进行region和word的对齐，在此基础上求出图片和句子的相似度。
    - #### 创新点
        先前的工作大多都是先求出图片中的每个region和句子中的每个word的相似度，然后将结果聚合起来作为图片和句子的相似度，
        但不同的区域或单词的重要性是不一样的。本文最大的创新/贡献就是引入了Stacked Cross Attention机制来考虑不同区域或单词的
        重要性。
    - #### Stacked Cross Attention
        本文给出了两种SCA范式：Image-Text SCA 和 Text-Image SCA，此处主要介绍 Image-Text Stacked Cross Attention.
        ![框架图]()
---