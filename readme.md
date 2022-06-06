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

