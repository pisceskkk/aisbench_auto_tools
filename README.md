# 项目简介
此项目旨在实现自动化工具，以简化某些操作。

## 前提条件
- Python 3.x
- 必备库（请参考requirements.txt）

## 安装步骤
1. 克隆仓库: `git clone https://github.com/pisceskkk/aisbench_auto_tools.git`
2. 进入项目目录: `cd aisbench_auto_tools`
3. 安装所需的依赖: `pip install -r requirements.txt`

## 使用说明
在使用前，请修改 `config.py` 文件以适配您的环境。

运行一下命令：
```bash
python3 aisbench_test.py --input_len 131066 --output_len 1 --data_num 64 --concurrency 2 --request_rate 0 --dataset_type prefix_cache --repeat_rate 99% --prefix_test --length_mean 65536 --length_std 32768 --length_min 4096 --length_max 131066
```

### CLI 参数说明
- `--input_len`: 输入长度
- `--output_len`: 输出长度
- `--data_num`: 数据数量
- `--concurrency`: 并发数
- `--request_rate`: 请求速率
- `--dataset_type`: 数据集类型
- `--repeat_rate`: 重复率
- `--prefix_test`: 前缀测试标志
- `--length_mean`: 长度均值
- `--length_std`: 长度标准差
- `--length_min`: 最小长度
- `--length_max`: 最大长度

## 注意事项
调整并发数和请求速率可帮助您调优系统性能。请根据提供的环境进行适当修改。
