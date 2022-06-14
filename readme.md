# CMeEE Project 2

## 运行依赖
详见requirements.txt。其中fastNLP安装可以参照https://fastnlp.readthedocs.io/zh/latest/user/installation.html

我们需要RoBERTa-large，以及来自于https://huggingface.co/uer/roberta-base-word-chinese-cluecorpussmall 的词级别RoEBRTa（如果需要使用词级别的话）。
请参照运行脚本并将对应的huggingface文件下载到指定目录。

## 运行最好结果
我们的最好结果来自于RoBERTa-large+30%数据增强。该配置运行脚本已被保存在`src/run_roberta_aug30.sh`。


