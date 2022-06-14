# CMeEE Project 2
> 郭奕玮、朱慕之、薛峥嵘
## 运行依赖
详见requirements.txt。其中fastNLP安装可以参照https://fastnlp.readthedocs.io/zh/latest/user/installation.html

我们需要RoBERTa-large，以及来自于https://huggingface.co/uer/roberta-base-word-chinese-cluecorpussmall 的词级别RoEBRTa（如果需要使用词级别的话）。
请参照运行脚本并将对应的huggingface文件下载到指定目录。

## 运行最好结果
我们的最好结果来自于RoBERTa-large+30%数据增强。该配置运行脚本已被保存在`src/run_roberta_aug30.sh`。
用到的增强过的数据在`data/CBLUEDatasets-aug30/CMeEE/`。

运行完成后，将会在`ckpts/roberta_large_crf_nested_2022_aug30/`下看到`CMeEE_test.json`。
然后运行`rule_for_dep.py`，将会在同目录下看到`CMeEE_test_updated_by_dep.json`。这即为最后用来评分的结果。
