"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse

from app.api.endpoints import router
from app.core.config import settings
from app.core.exceptions import AssetValidationError
from app.core.logging import get_logger
from app.services.job_inference import ModelInferenceService, PredictionJobManager

logger = get_logger(__name__)

BACKEND_GUIDE_HTML = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>桥梁智能评估系统 API 指南</title>
    <style>
      body {
        margin: 0;
        font-family: "Segoe UI", "PingFang SC", sans-serif;
        background: linear-gradient(180deg, #f7fbfc 0%, #e8f1f4 100%);
        color: #22333b;
      }
      .page {
        max-width: 1080px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }
      .hero, .card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(34,51,59,0.08);
        border-radius: 22px;
        box-shadow: 0 20px 50px rgba(22,37,44,0.08);
      }
      .hero {
        padding: 28px;
        margin-bottom: 20px;
      }
      .cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 16px;
      }
      .card {
        padding: 20px;
      }
      h1 { margin: 0 0 12px; }
      h2 { margin: 0 0 10px; font-size: 1.05rem; }
      p, li { line-height: 1.7; }
      ul { padding-left: 20px; margin: 10px 0 0; }
      .links {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 16px;
      }
      a.button {
        text-decoration: none;
        padding: 10px 14px;
        border-radius: 14px;
        background: linear-gradient(135deg, #0b6e4f, #005f73);
        color: white;
        font-weight: 700;
      }
      code {
        padding: 2px 6px;
        border-radius: 8px;
        background: rgba(0,95,115,0.08);
      }
      .tip {
        margin-top: 16px;
        padding: 12px 14px;
        border-left: 4px solid #0b6e4f;
        background: rgba(11,110,79,0.08);
        border-radius: 12px;
      }
    </style>
  </head>
  <body>
    <div class="page">
      <section class="hero">
        <p>Bridge Intelligent Assessment API</p>
        <h1>桥梁智能评估系统后端说明</h1>
        <p>
          你现在看到的是后端服务首页，不是预测大屏。前端可视化界面在
          <code>http://127.0.0.1:5173</code>。
          后端页面主要用于查看接口说明、测试上传文件、检查模型是否加载成功。
        </p>
        <div class="links">
          <a class="button" href="/docs">接口文档 / Swagger Docs</a>
          <a class="button" href="/redoc">接口说明 / ReDoc</a>
          <a class="button" href="/api/v1/health">健康检查 / Health</a>
          <a class="button" href="/api/v1/models">模型状态 / Models</a>
        </div>
        <div class="tip">
          Swagger UI 本身的按钮和系统控件默认是英文，这是它的前端框架行为；
          我已经把接口标题、说明、字段语义改成中文/双语，便于阅读。
        </div>
      </section>
      <section class="cards">
        <article class="card">
          <h2>页面含义</h2>
          <ul>
            <li><code>/docs</code>：在线接口测试页，可直接上传 Excel/CSV/JSON。</li>
            <li><code>/redoc</code>：更适合阅读的接口说明页。</li>
            <li><code>/api/v1/health</code>：看服务和模型是否正常加载。</li>
            <li><code>/api/v1/models</code>：看 5 个基模型和 scaler 是否已就绪。</li>
          </ul>
        </article>
        <article class="card">
          <h2>推荐使用方式</h2>
          <ul>
            <li>普通使用：打开前端大屏 <code>http://127.0.0.1:5173</code></li>
            <li>开发调试：打开 <code>/docs</code> 手动测试接口</li>
            <li>查看说明：打开 <code>/redoc</code> 阅读接口定义</li>
          </ul>
        </article>
        <article class="card">
          <h2>当前接口</h2>
          <ul>
            <li><code>POST /api/v1/predict/file</code> 文件上传推理</li>
            <li><code>POST /api/v1/predict/json</code> JSON 数据推理</li>
            <li><code>GET /api/v1/health</code> 服务状态</li>
            <li><code>GET /api/v1/models</code> 模型资产状态</li>
          </ul>
        </article>
      </section>
    </div>
  </body>
</html>
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="桥梁智能评估系统 API / Bridge Intelligent Assessment API",
        summary="桥梁行车安全智能预测与评价后端服务",
        description=(
            "用于桥梁行车安全预测、区间不确定性解耦和多级预警的后端接口。\n\n"
            "This backend provides file/JSON inference APIs, model health checks, and"
            " uncertainty-aware bridge assessment outputs."
        ),
        version=settings.model_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = ModelInferenceService()
    app.state.inference_service = service
    app.state.job_manager = PredictionJobManager(service)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(BACKEND_GUIDE_HTML)

    @app.on_event("startup")
    def startup_event() -> None:
        try:
            service.load_assets()
        except AssetValidationError:
            logger.exception("Asset validation failed during startup.")
            raise

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            summary=app.summary,
            description=app.description,
            routes=app.routes,
        )
        for path_item in openapi_schema.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation.get("responses", {}).pop("422", None)
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()
