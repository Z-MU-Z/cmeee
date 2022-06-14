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
我们在仓库中已经包含了这个结果，但是命名为`ckpts/roberta_large_crf_nested_2022_aug30/update_by_dep/CMeEE_test.json`。

我们还提供了测试脚本`src/eval_from_json.py`。其可以对两个json进行比较并输出评测指标。
运行之还会在src文件夹下产生`result.html`，是我们在实验中用于观察的重要手段。

src文件夹下还包含了其余运行脚本以供参考。