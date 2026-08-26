# 各阶段验证代码片段模板

直接套用，按理解文档中的具体数值替换占位符。

## ① 数据处理验证

```python
# 取原始样本与处理后样本对比
raw_sample = raw_data[0]
processed_sample = preprocess_fn(raw_sample)
print("处理前:", raw_sample)
print("处理后:", processed_sample)
# 人工核对：是否符合理解文档 1.4 描述的步骤
```

## ② 数据导入验证

```python
loader = PaperDataLoader(config).train_loader()
batch = next(iter(loader))

# 替换为理解文档 2.1 记录的实际 shape
expected_shape = (config.batch_size, ...)
assert batch[0].shape == expected_shape, f"实际 shape {batch[0].shape} 与理解文档不符"
print("dtype:", batch[0].dtype)
print("✅ 数据导入验证通过")
```

## ③ 模型结构验证

```python
model = PaperModel(config, ...)
model.set_eval()

x = batch[0]
with torch.no_grad():
    out = model._net(x) if hasattr(model, "_net") else model.forward(x)

expected_out_shape = (...)  # 对照理解文档 2.2
assert out.shape == expected_out_shape, f"输出 shape {out.shape} 与理解文档不符"
print("✅ 模型结构验证通过，输出 shape:", out.shape)
```

逐层验证（如果结构较复杂，建议在 forward 内临时加 print 逐层确认，调试完成后删除）：

```python
def forward(self, x):
    x = self.layer1(x); print("layer1 out:", x.shape)
    x = self.layer2(x); print("layer2 out:", x.shape)
    return x
```

## ④ 模型评估验证

```python
# 构造一个手算好预期结果的小样本
fake_predictions = [...]
fake_targets = [...]
expected_metric = ...  # 手算的预期值

result = compute_metrics(fake_predictions, fake_targets, k=20)
assert abs(result["recall@20"] - expected_metric) < 1e-4, \
    f"评估函数计算值 {result} 与手算值 {expected_metric} 不一致"
print("✅ 评估函数验证通过")
```

## ⑤ 模型训练验证（单 batch / 单 epoch）

```python
model.set_train()
param_before = next(model.parameters()).clone()

metrics = model.train_step(batch)

assert not torch.isnan(torch.tensor(metrics["loss"])), "loss 出现 NaN"
assert not torch.isinf(torch.tensor(metrics["loss"])), "loss 出现 Inf"
assert metrics["loss"] < 1e4, f"loss 数值异常偏大: {metrics['loss']}"

param_after = next(model.parameters())
assert not torch.equal(param_before, param_after), "参数没有更新，检查反向传播/优化器"

# 【必检】train_step 所有返回值必须为标量（float/int 或 0维张量）
for k, v in metrics.items():
    assert isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1), \
        f"train_step 指标 '{k}' 不是标量: type={type(v)}, value={v}"
print("train_step scalars OK")

print("训练 step 验证通过，loss:", metrics["loss"])

# ── eval_step 标量契约验证（必须执行） ──
# 防止 eval_step 返回大批量张量导致 Trainer.run() 中 sum(v) 对不同 batch 形状 crash
model.set_eval()
with torch.no_grad():
    eval_metrics = model.eval_step(val_batch)

for k, v in eval_metrics.items():
    assert isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1), \
        f"eval_step 指标 '{k}' 不是标量！Trainer 的 sum(v)/len(v) 聚合会在尾批尺寸不同时 crash。" \
        f"type={type(v)}, value={v}"
print("eval_step scalars OK")
```

## ⑤-b 不等长尾批压测（推荐系统/攻击模式必检）

```python
# 用 drop_last=False 的 DataLoader 产生不等长尾批
# 模拟 batch_size 不整除数据集大小的场景
loader_uneven = TorchDataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
batch_sizes = []
for batch in loader_uneven:
    batch_sizes.append(len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch))

assert len(set(batch_sizes)) > 1, \
    f"所有 batch 尺寸相同 {set(batch_sizes)}，无法压测不等长情况。" \
    f"尝试调整 batch_size 使不整除数据集大小"
print(f"不等长尾批压测通过: batch 尺寸分布 = {set(batch_sizes)}")

# 在此 loader 上跑一个 eval epoch，确认不 crash
model.set_eval()
with torch.no_grad():
    for batch in loader_uneven:
        _ = model.eval_step(batch)
print("不等长尾批 eval 压测通过")
```

## ⑥ 结果展示验证

```python
def compare_with_paper(reproduced: dict, reported: dict, tolerance: float = 0.02):
    print(f"{'指标':<15}{'论文报告值':<15}{'复现值':<15}{'相对差距':<10}{'判定'}")
    for k in reported:
        rep_val = reported[k]
        rep_actual = reproduced.get(k, float("nan"))
        diff = abs(rep_actual - rep_val) / max(abs(rep_val), 1e-8)
        verdict = "完全对齐" if diff <= tolerance else "需排查"
        print(f"{k:<15}{rep_val:<15.4f}{rep_actual:<15.4f}{diff:<10.2%}{verdict}")
```
