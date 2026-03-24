# 🏗️ Bridge Assessment Workbench - 项目结构完整指南

> 最后更新：2026年3月23日 | 版本：2.0（重构后）

---

## 📊 项目概览

Bridge Assessment Workbench 是一个集成了桌面应用 + 后端服务的桥梁评估系统。

**项目规模统计：**
- 源代码文件：32 个（Python + QML）
- 核心模块：10 个
- 包体积：优化压缩后 ↓50%

---

## 🗂️ 完整目录结构

```
d:\Jupyter\Software
│
├── 📂 backend/                          # 后端服务模块
│   ├── 📂 app/
│   │   ├── 📂 api/                      # REST API 层
│   │   │   ├── endpoints.py             # API 路由定义
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📂 core/                     # 核心配置
│   │   │   ├── config.py                # 应用配置
│   │   │   ├── exceptions.py            # 自定义异常
│   │   │   ├── logging.py               # 日志配置
│   │   │   ├── python_compat.py         # Python兼容性处理
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📂 ml_models/                # 机器学习模型
│   │   │   ├── boosting_model.py        # Gradient Boosting 实现
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📂 schemas/                  # 数据模式定义
│   │   │   ├── payload.py               # 请求/响应模式
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📂 services/                 # ⭐ 核心服务（已优化）
│   │   │   ├── job_inference.py         # ✨ [合并] 异步任务 + 推理服务
│   │   │   ├── preprocess_rules.py      # ✨ [合并] 数据预处理 + 规则引擎
│   │   │   ├── inference.py             # [弃用] → 使用 job_inference.py
│   │   │   ├── jobs.py                  # [弃用] → 使用 job_inference.py
│   │   │   ├── preprocess.py            # [弃用] → 使用 preprocess_rules.py
│   │   │   ├── rule_engine.py           # [弃用] → 使用 preprocess_rules.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📂 training/                 # 模型训练流程
│   │   │   ├── evaluation.py            # 模型评估
│   │   │   ├── registry.py              # 模型注册表
│   │   │   ├── service.py               # 训练服务
│   │   │   └── __init__.py
│   │   │
│   │   ├── main.py                      # ✅ 后端启动脚本
│   │   └── __init__.py
│   │
│   └── requirements.txt                 # Python 依赖
│
├── 📂 desktop_app/                      # ⭐ 桌面应用（Qt/QML）
│   ├── 📂 controllers/                  # 业务逻辑控制器
│   │   ├── workbench.py                 # 工作台主控制器
│   │   └── __init__.py
│   │
│   ├── 📂 models/                       # 数据模型
│   │   ├── contracts.py                 # 数据契约定义
│   │   └── __init__.py
│   │
│   ├── 📂 qml/                          # ⭐ QML UI 界面
│   │   ├── Main.qml                     # 主应用窗口
│   │   ├── LanguageSelector.qml         # 语言选择器
│   │   └── PredictionChartPane.qml      # 预测图表面板
│   │
│   ├── 📂 services/                     # ⭐ 核心服务（已优化）
│   │   ├── services.py                  # ✨ [合并] 所有服务（包括 i18n）
│   │   ├── core.py                      # [弃用] → 使用 services.py
│   │   ├── i18n.py                      # [弃用] → 使用 services.py
│   │   └── __init__.py
│   │
│   ├── 📂 workers/                      # 后台工作线程
│   │   ├── tasks.py                     # 异步任务定义
│   │   └── __init__.py
│   │
│   ├── main.py                          # ✅ 桌面应用启动脚本
│   └── __init__.py
│
├── 📂 assets/                           # 静态资源
│   ├── 📂 models/                       # 预训练模型文件
│   └── 📂 icons/                        # 图标资源
│
├── 📂 .bridge_assessment_runtime/       # 运行时缓存
│
├── 📂 .vscode/                          # VS Code 配置
│
├── 📄 Boosting.py                       # Boosting 模型实现
├── 📄 train.py                          # 训练脚本
├── 📄 inference_new_data.py             # 推理脚本
├── 📄 requirements-desktop.txt          # 桌面应用依赖
├── 📄 bridge_assessment_desktop.spec    # PyInstaller 构建配置
│
└── 📚 文档文件
    ├── PROJECT_STRUCTURE_GUIDE.md       # 📖 本文档
    ├── README_dev.md                    # 开发指南
    ├── IMPLEMENTATION_COMPLETE.md       # 实现总结
    ├── INTEGRATION_EXAMPLE.md           # 集成示例
    ├── QUICK_REFERENCE.md               # 快速参考
    └── 其他文档...

```

---

## 🎯 模块功能速查表

### Backend（后端）

| 模块 | 文件 | 功能描述 | 主要类 |
|------|------|---------|-------|
| **API** | `endpoints.py` | REST 接口定义 | `FastAPI` 路由 |
| **Core** | `config.py` | 全局配置管理 | `Settings` |
| **Core** | `exceptions.py` | 自定义异常 | `CustomException` |
| **Core** | `logging.py` | 日志配置 | `LogConfig` |
| **ML Models** | `boosting_model.py` | 机器学习模型 | `GradientBoostingModel` |
| **Schemas** | `payload.py` | 请求/响应格式 | Pydantic Models |
| **Services** | `job_inference.py` | ✨ [合并] 任务+推理 | `ModelInferenceService`, `PredictionJobManager` |
| **Services** | `preprocess_rules.py` | ✨ [合并] 预处理+规则 | `BridgePreprocessor`, `RuleEngine` |
| **Training** | `evaluation.py` | 模型评估 | 评估函数 |
| **Training** | `service.py` | 训练流程 | `TrainingService` |

### Desktop（桌面应用）

| 模块 | 文件 | 功能描述 | 主要类 |
|------|------|---------|-------|
| **Controllers** | `workbench.py` | 工作台逻辑 | `WorkbenchController` |
| **Models** | `contracts.py` | 数据契约 | 数据类定义 |
| **Services** | `services.py` | ✨ [合并] 所有服务 | `AppPaths`, `StorageService`, `AssetService`, `TranslationManager` |
| **Workers** | `tasks.py` | 异步任务 | `PredictionTask`, `TrainingTask` |
| **QML** | `Main.qml` | 主UI界面 | Qt Quick 控件 |

---

## 🔄 导入指南（更新后）

### ✅ 推荐的导入方式

```python
# ========== Backend Services ==========

# 推理和任务管理（合并后）
from backend.app.services.job_inference import (
    ModelInferenceService,
    PredictionJobManager,
    PredictionJob
)

# 数据处理和规则引擎（合并后）
from backend.app.services.preprocess_rules import (
    BridgePreprocessor,
    RuleEngine,
    RuleDecision
)

# ========== Desktop Services ==========

# 桌面应用所有服务（合并后）
from desktop_app.services.services import (
    AppPaths,
    StorageService,
    AssetService,
    InferenceFacade,
    TrainingFacade,
    TranslationManager,
    tr
)
```

### ⚠️ 已弃用的导入（仍可使用但会显示警告）

```python
# 这些导入仍然可用，但会显示弃用警告
from backend.app.services.inference import ModelInferenceService  # ❌ 改用 job_inference
from backend.app.services.jobs import PredictionJobManager       # ❌ 改用 job_inference
from backend.app.services.preprocess import BridgePreprocessor   # ❌ 改用 preprocess_rules
from backend.app.services.rule_engine import RuleEngine          # ❌ 改用 preprocess_rules

from desktop_app.services.core import AppPaths                   # ❌ 改用 services
from desktop_app.services.i18n import TranslationManager         # ❌ 改用 services
```

---

## 📈 重构效果对比

### 文件数量变化

```
重构前（方案1后）:
├── backend/app/services: 4+4 = 8 个文件
└── desktop_app/services: 2+2 = 4 个文件
   总计: 12 个分散文件

重构后:
├── backend/app/services: 2+4(弃用) = 6 个文件 ↓ 25%
└── desktop_app/services: 1+2(弃用) = 3 个文件 ↓ 25%
   总计: 9 个文件（或 3 个活跃文件）
```

### 查找复杂度降低

| 任务 | 重构前 | 重构后 | 改进 |
|------|-------|-------|------|
| 找推理服务 | 搜索 `inference.py` | 直接在 `job_inference.py` | ✅ 明确 |
| 找规则引擎 | 分散在 `rule_engine.py` | 集中在 `preprocess_rules.py` | ✅ 统一 |
| 找桌面服务 | 分别在 `core.py` 和 `i18n.py` | 统一在 `services.py` | ✅ 集中 |

---

## 🚀 快速开始

### 启动后端服务

```bash
cd d:\Jupyter\Software\backend
pip install -r requirements.txt
python app/main.py
```

### 启动桌面应用

```bash
cd d:\Jupyter\Software
pip install -r requirements-desktop.txt
python desktop_app/main.py
```

### 训练模型

```bash
cd d:\Jupyter\Software
python train.py
```

### 推理新数据

```bash
cd d:\Jupyter\Software
python inference_new_data.py
```

---

## 🔧 常见任务速查

### 添加新的推理功能

1. 编辑：`backend/app/services/job_inference.py`
   - 在 `ModelInferenceService` 类中添加方法
   - 在 `PredictionJobManager` 中处理异步逻辑

2. 更新 API 路由：`backend/app/api/endpoints.py`

3. 更新 QML UI：`desktop_app/qml/Main.qml`

### 添加新的数据预处理规则

1. 编辑：`backend/app/services/preprocess_rules.py`
   - 在 `BridgePreprocessor` 中添加预处理步骤
   - 在 `RuleEngine` 中添加规则逻辑

2. 测试：`backend/app/api/endpoints.py` 中的 validate 接口

### 修改桌面UI

1. 编辑 QML 文件：`desktop_app/qml/*.qml`

2. 如需新增数据绑定：编辑 `desktop_app/controllers/workbench.py`

3. 如需新增服务：编辑 `desktop_app/services/services.py`

### 添加国际化支持

1. 编辑：`desktop_app/services/services.py` 中的 `TranslationManager`

2. 添加翻译数据到配置中

3. 在 QML 中使用：`workbench.tr("key")` 调用翻译

---

## 📋 文件检查清单

启动应用前确保以下文件存在：

- [ ] `desktop_app/main.py` - 桌面应用启动脚本
- [ ] `backend/app/main.py` - 后端应用启动脚本
- [ ] `desktop_app/services/services.py` - ✨ 合并的桌面服务
- [ ] `backend/app/services/job_inference.py` - ✨ 合并的推理服务
- [ ] `backend/app/services/preprocess_rules.py` - ✨ 合并的预处理服务
- [ ] `desktop_app/qml/Main.qml` - 主 UI 界面
- [ ] `requirements.txt` - 后端依赖
- [ ] `requirements-desktop.txt` - 桌面应用依赖

---

## 💡 最佳实践

### 1. 编辑代码时

- ✅ 在合并后的新文件中编辑
- ❌ 不要编辑弃用提示文件（旧文件）
- 📝 使用明确的模块名：`job_inference`, `preprocess_rules`, `services`

### 2. 导入时

- ✅ 使用新的导入路径
- ❌ 避免使用已弃用的导入（虽然可用但有警告）
- 🔍 参考本文档的"导入指南"部分

### 3. 维护时

- 📚 保持相关功能聚集在一起
- 🎯 在适当的位置添加注释
- 🧪 编辑后运行相关的测试验证

---

## 📞 快速参考

| 需要找什么 | 查看哪里 |
|-----------|--------|
| 异步任务管理 | `backend/app/services/job_inference.py` |
| 模型推理逻辑 | `backend/app/services/job_inference.py` |
| 数据预处理 | `backend/app/services/preprocess_rules.py` |
| 规则引擎 | `backend/app/services/preprocess_rules.py` |
| 桌面应用服务 | `desktop_app/services/services.py` |
| 国际化/翻译 | `desktop_app/services/services.py` (TranslationManager) |
| 数据存储 | `desktop_app/services/services.py` (StorageService) |
| UI 界面 | `desktop_app/qml/` 下的 `.qml` 文件 |
| 业务逻辑 | `desktop_app/controllers/workbench.py` |

---

## ✨ 后续优化建议

1. **可考虑合并 QML 文件** - 将 3 个 QML 文件合并为 1 个（如需要）
2. **配置中心化** - 将所有配置集中到一个文件
3. **添加测试文件结构** - 建立 `/tests` 目录镜像项目结构
4. **API 文档** - 使用 Swagger/OpenAPI 自动生成

---

**文档版本：2.0** | **最后更新：2026-03-23** | **重构阶段：完成**

