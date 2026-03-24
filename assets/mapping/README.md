# 映射模型目录

这里用于放置“桥-轨映射”阶段的轻量化模型，例如：
- `bridge_track_mapping.joblib`
- `bridge_track_mapping.pkl`
- `bridge_track_mapping.pt`
- `bridge_track_mapping.pth`

当前代码已经支持“映射模型可选”。
- 如果目录里存在上述模型，后端会在启动时自动加载。
- 如果目录为空，`raw_measurements` 模式会先做 WCSS 样条重构，再把平滑后的曲线直接作为 `track_deformation` 占位输入，并将三类不平顺默认置零。
- 这能保证工程先跑通，但不等价于论文中的完整桥-轨映射流程；后续你只需要把真实映射模型文件放进本目录，无需改代码。
