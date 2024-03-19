import argparse
import asyncio
import importlib
import inspect
import json
import os
import ssl
from contextlib import asynccontextmanager
from http import HTTPStatus

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import LoRA
from vllm.logger import init_logger

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    # 这是Python 3.7及更高版本中定义异步函数的方式。
    # 异步函数可以包含一个或多个await表达式，它们允许函数暂停并在等待其他异步操作完成后继续执行。
    async def _force_log():
        while True:
            # 这个表达式让当前运行的协程暂停10秒钟。这有助于防止CPU占用过高，并确保任务在后台以非阻塞的方式执行。
            await asyncio.sleep(10)
            # 此语句调用 engine 对象的 do_log_stats 方法。
            # 由于该方法是异步的，使用 await 关键字等待其完成。
            # 这意味着 _force_log 函数会在 do_log_stats 方法执行完毕后继续到下一次循环。
            await engine.do_log_stats()

    # 检查一个条件，如果disable_log_stats不是True，那么它会创建一个新的任务去运行 _force_log。
    # 这意味着do_log_stats将在后台非阻塞地定期执行，只在engine_args配置要求的情况下才会启用。
    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())
    # 语句是异步上下文管理器的关键部分。
    # 当进入上下文管理器（例如，使用async with lifespan(app):）时，
    # yield后面的代码将被执行。
    # 在这个例子中，_force_log任务可能已经在后台开始运行，而yield语句允许其他代码继续执行。
    # 当离开上下文管理器时（async with块结束），异步上下文管理器的清理工作通常在这里进行，但在这个例子中，没有显式的清理操作。
    yield


# 这段代码是在创建FastAPI应用实例时，将lifespan上下文管理器作为参数传递。
# FastAPI(lifespan=lifespan)使得lifespan中的异步函数在应用程序的生命周期中起作用。
# 在FastAPI中，lifespan用于控制应用的启动、运行和关闭过程。
# 通过设置lifespan，你可以自定义这些阶段的行为，例如执行初始化任务、清理资源等。
# 在这个例子中，lifespan是一个异步上下文管理器，它会在应用启动时创建一个后台任务来定期执行日志统计（如果未被禁用）。
# 这使得 _force_log 在FastAPI应用开始运行时启动，并在应用关闭前一直运行，除非你在配置中禁用了日志统计。
# 这样的话，engine.do_log_stats() 将会在后台以非阻塞的方式每10秒执行一次。
app = fastapi.FastAPI(lifespan=lifespan)


class LoRAParserAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        lora_list = []
        for item in values:
            name, path = item.split("=")
            lora_list.append(LoRA(name, path))
        setattr(namespace, self.dest, lora_list)


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical", "trace"],
        help="log level for uvicorn",
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="If provided, the server will require this key "
        "to be presented in the header.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the huggingface name.",
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        default=None,
        nargs="+",
        action=LoRAParserAction,
        help="LoRA module configurations in the format name=path. "
        "Multiple modules can be specified.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="The file path to the chat template, "
        "or the template in single-line form "
        "for the specified model",
    )
    parser.add_argument(
        "--response-role",
        type=str,
        default="assistant",
        help="The role name to return if " "`request.add_generation_prompt=true`.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="The file path to the SSL key file",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="The file path to the SSL cert file",
    )
    parser.add_argument(
        "--ssl-ca-certs", type=str, default=None, help="The CA certificates file"
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, vLLM will add it to the server "
        "using @app.middleware('http'). "
        "If a class is provided, vLLM will add it to the server "
        "using app.add_middleware(). ",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()

    # 加入默认的中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if not request.url.path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    # 加入参数传递的中间件
    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. " f"Must be a function or a class."
            )

    logger.info(f"vLLM API server version {vllm.__version__}")
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    # 解析引擎参数
    # 从命令行参数中创建AsyncEngineArgs对象
    engine_args = AsyncEngineArgs.from_cli_args(args)
    # 创建引擎
    # 从引擎参数创建AsyncLLMEngine对象
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # 创建OpenAIServingChat对象
    # 从引擎、模型名称、响应角色、LoRA模块和聊天模板创建OpenAIServingChat对象
    # 该对象用于处理聊天请求
    openai_serving_chat = OpenAIServingChat(
        engine, served_model, args.response_role, args.lora_modules, args.chat_template
    )
    # 创建OpenAIServingCompletion对象
    # 从引擎、模型名称和LoRA模块创建OpenAIServingCompletion对象
    # 该对象用于处理completion生成请求
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model, args.lora_modules
    )

    app.root_path = args.root_path
    # 启动FastAPI应用，使用uvicorn作为异步Web服务器

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
