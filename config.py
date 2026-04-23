# 数据集文件夹
DATASET_PATH = "PATH/TO/DATASET"
# aisbench 工作路径, 为 git clone aisbench 后得到的 benchmark 目录的绝对路径
# 可通过命令 `pip show ais-bench-benchmark | grep location` 查询
WORK_PATH = "PATH/TO/aisbench/benchmark"
# 服务化配置的模型名称
MODEL_NAME = "MODEL_NAME"
# 模型权重路径, 用于读取 tokenizer
MODEL_PATH = "PATH/TO/MODEL/WEIGHTS"
# 请求目的 IP
HOST_IP = "localhost"
# 请求目的端口
HOST_PORT = "8000"

# 如果使用稳态测试请将该字段设置为 "stable_stage"
DEFAULT_PERFORMANCE_TEST = "default_perf"