# pairwise-vertex-decode-quantize

## 领域

游戏、图形与多媒体

## 主题

顶点量化与索引边界

## 说明

分块量化、子网格索引与解码后绑定。

初始状态下 `graphics/vertex_decode_quantize.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
