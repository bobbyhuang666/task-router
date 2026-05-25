# Changelog

All notable changes to TaskRouter will be documented in this file.

## [4.5.0] - 2026-05-28

### Added
- API 认证中间件：支持 Bearer token、X-API-Key header、query 参数三种方式
- `TASKROUTER_API_KEY` 环境变量控制认证（为空则开放访问）
- Python logging 框架：`logger.py` 统一日志配置，`TASKROUTER_LOG_LEVEL` 环境变量
- API 请求日志：任务执行、OpenAI 兼容端点记录 route/model/耗时
- 认证失败日志：记录来源 IP 和请求路径
- 健康检查和仪表盘页面无需认证（PUBLIC_PATHS 白名单）
- CORS 中间件新增 Authorization 和 X-API-Key header 支持

### Changed
- `api_server.py` 版本号更新为 4.5.0
- 启动信息显示认证状态

## [4.4.1] - 2026-05-28

### Fixed
- P0: `read_jsonl` not imported in `task_router.py` causing NameError on `--stats` and adaptive thresholds

### Added
- Regression tests for `show_usage_stats` and `CapabilityTracker`

## [4.4.0] - 2026-05-28

### Fixed
- `quality_eval.py`: `detect_task_type` missing required `templates` argument
- `quality_eval.py`: fragile `dir()` check replaced with proper `None` flag
- `privacy.py`: removed shadowed first `get_privacy_filter` definition
- `api_server.py`: hardcoded version "2.0" updated to match actual version

### Added
- `io_utils.py`: shared JSONL read/write/append utilities
- YAML/JSON config file auto-discovery (`config.yaml`, `config.json`)
- 19 new tests: cache concurrency, OpenAI message parsing, run_task integration, config loading

### Changed
- Consolidated JSONL I/O across 6 files to use `io_utils`
- Unified hardcoded `~/.cache/task_router` paths to `get_config().cache_dir`
- Simplified `show_usage_stats` with `_classify_route` helper
- `pyproject.toml` `pythonpath` setting replaces `sys.path.insert` hack in tests

### Removed
- `local_agent.py` (duplicated `models.call_ollama`)
- 11 unused imports, 1 unused variable

## [4.3.0] - 2026-05-28

### Added
- Cache concurrent safety tests (multi-thread stress)
- OpenAI endpoint message parsing tests (7 cases)
- `run_task` integration tests with mock Ollama
- Config loading tests (JSON, missing, invalid)
- `conftest.py` for test infrastructure

### Fixed
- `CircuitBreaker.record_success()` now resets `failures` counter
- `_run_local_subtasks` correctly tracks `rule_engine` vs model usage
- Cost savings display clarified as "theoretical max"

## [4.2.0] - 2026-05-28

### Added
- CircuitBreaker HALF_OPEN state: CLOSED → OPEN → HALF_OPEN → CLOSED
- Distillation TTL/forgetting mechanism (90d default, 14d contested, 7d outdated)
- `cleanup_expired()` with FIFO cap at 5000 entries
- `--distill-cleanup` CLI command
- 16 new tests: circuit breaker states (7), distillation TTL (7), cache TTL (2)

### Changed
- `run_task()` split into 7 focused sub-functions
- `call_cloud_api` uses `allow_request()` for half-open probing

## [4.1.0] - 2026-05-28

### Fixed
- Cache `set()` race condition: disk I/O moved inside lock
- `ModelRegistry` singleton with thread-safe double-checked locking
- Unified `PrivacyFilter` singleton across modules
- OpenAI compatible API message parsing improvements

### Added
- Output validation + cloud fallback
- Semantic cache (trigram Jaccard)
- Dynamic few-shot selection from distillation pool
- RLM-style recursive task decomposition
- Adaptive confidence thresholds per capability
