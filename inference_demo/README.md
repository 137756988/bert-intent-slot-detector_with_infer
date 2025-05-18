# 简化版金融意图识别工具

这是一个简化版的金融意图识别工具，可以识别用户输入文本的意图和领域。只保留了核心的推理功能，便于集成到其他系统中。

## 文件说明

- `infer.py` - 简化版推理脚本，提供基本的意图识别功能

## 使用方法

### 命令行参数

脚本支持以下命令行参数：

- `--model_dir` - 模型目录路径，默认为 `output_model/FinQA_roberta_wwm_ext_large_20250518_v1`
- `--use_gpu` - 使用GPU进行推理（如果可用）
- `--text` - 要分析的文本，如不提供则进入交互模式

### 交互模式

直接运行脚本，进入交互式意图识别模式：

```bash
python inference_demo/infer.py
```

在交互模式下：
- 输入任意文本进行意图识别
- 输入 `exit` 或 `quit` 退出程序

### 单次查询模式

如果只需分析一条文本，可以使用 `--text` 参数：

```bash
python inference_demo/infer.py --text "什么是市盈率"
```

### 示例输出

```
$ python inference_demo/infer.py --text "什么是市盈率"
正在加载模型...
模型已加载，使用设备: cpu
模型路径: .../model/model_epoch4
意图标签数量: 3
槽位标签数量: 9
模型加载完成!
{
  "text": "什么是市盈率",
  "domain": "finance",
  "intent": "KNOWLEDGE_QUERY",
  "slots": {
    "keyword": ["市盈率"]
  }
}
```

## 预测结果格式

预测结果为JSON格式，包含以下字段：

- `text` - 输入的文本
- `domain` - 领域（固定为"finance"）
- `intent` - 识别出的意图，可能的值为：
  - `KNOWLEDGE_QUERY` - 金融知识咨询
  - `STOCK_ANALYSIS` - 股票分析
- `slots` - 识别出的槽位信息（可选）
  - `keyword` - 咨询的金融术语关键词
  - `stock_name` - 股票名称
  - `stock_code` - 股票代码

## 与其他系统集成

如需在其他系统中集成该工具，可以直接导入并调用函数：

```python
from inference_demo.infer import load_model, predict_intent

# 加载模型
detector = load_model("output_model/FinQA_roberta_wwm_ext_large_20250518_v1", use_gpu=False)

# 预测意图
text = "解释一下什么是股息率"
result = predict_intent(detector, text)
print(result)

# 判断意图类型
if result["intent"] == "KNOWLEDGE_QUERY":
    # 处理金融知识咨询
    pass
elif result["intent"] == "STOCK_ANALYSIS":
    # 处理股票分析
    pass
```

## 注意事项

1. 首次运行时会从模型目录加载最新的模型，可能需要一定时间
2. 目前支持判断两种主要意图：金融知识查询和股票分析
3. 默认使用CPU进行推理，如需更快速度可添加 `--use_gpu` 参数 