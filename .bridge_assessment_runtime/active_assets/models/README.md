# 资产说明

当前目录用于放置后端推理时必须加载的模型资产。

已整理进来的文件：
- `ensemble_model_1.pth` ~ `ensemble_model_5.pth`
- `scaler_y.pkl`

仍然建议补齐的文件：
- `scaler_X.pkl`：若输入不是已经标准化后的特征，则后端需要它来复现训练时的特征缩放。
- `bridge_track_mapping.joblib` 或等价映射模型：用于从桥梁重构线形映射到轨道激扰。
- `feature_columns.json`：固定推理时的特征顺序，避免列顺序错误。
