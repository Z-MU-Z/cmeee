# CMeEE Project 1


代码运行方式与助教提供的无异。即更改`run_cmeee2.sbatch`中的`TASK_ID`，来选择是否用嵌套处理、是否CRF。

---

参考CBLUE上的测试结果：

| 模型            | F1-score |
| --------------- | -------- |
| Linear          | 0.62174  |
| CRF             | 0.62417  |
| Linear+嵌套处理 | 0.62279  |
| CRF+嵌套处理    | 0.63063  |

不同encoder的比较

均为 CRF + Nested

| Encoder                                       | Score       |
| --------------------------------------------- | ----------- |
| bert-base-chinese                             | 0.630633683 |
| 9pinus/macbert-base-chinese-medical-collation | 0.600446971 |
| trueto/medbert-base-chinese                   | 0.631047539 |
| clue/roberta_chinese_large                    | 0.643820388 |
| clue/roberta_chinese_base                     | 0.635276496 |




Prompt tuning完全没用
| token |             |
| ----- | ----------- |
| 0     | 0.516107588 |
| 1     | 0.507114736 |
| 5     | 0.449615385 |
| 10    | 0.377838661 |
| 20    | 0.001571256 |
