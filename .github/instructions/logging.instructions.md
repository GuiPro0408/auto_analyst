---
applyTo: "**"
---

# Logging Guidelines

## Overview

Auto-Analyst uses a centralized logging system with **run correlation IDs** for tracing requests through the pipeline. All logging is configured in `api/logging_setup.py`.

## Getting a Logger

Always use `get_logger()` from `api/logging_setup`:

```python
from api.logging_setup import get_logger

# Without run correlation (module-level logging)
logger = get_logger(__name__)

# With run correlation (inside pipeline nodes)
log = get_logger("api.graph.plan", run_id=state.get("run_id"))
```

**Never use `logging.getLogger()` directly** — it bypasses the centralized configuration.

## Run Correlation IDs

Every research run gets a UUID (`run_id`) that flows through all pipeline stages:

```python
# In run_research()
run_id = str(uuid4())
initial_state = {"query": query, "run_id": run_id, ...}

# In each node
log = get_logger("api.graph.fetch", run_id=state.get("run_id"))
log.info("fetch_complete", extra={"documents": len(documents)})
```

Output includes `[run=<uuid>]` for correlating logs across nodes:

```
2025-01-15 10:30:45 INFO [api.graph.plan] [run=abc-123] plan_complete
2025-01-15 10:30:46 INFO [api.graph.search] [run=abc-123] search_complete
```

## Logging Pattern in Pipeline Nodes

Follow this pattern for consistent observability:

```python
def my_node(state: Dict):
    log = get_logger("api.graph.my_node", run_id=state.get("run_id"))
    start = perf_counter()

    try:
        result = do_work()
        log.info("my_node_complete", extra={
            "result_count": len(result),
            "duration_ms": (perf_counter() - start) * 1000
        })
        return {"output": result}
    except Exception as exc:
        log.exception("my_node_failed")
        errors = state.get("errors", [])
        errors.append(f"my_node_failed: {exc}")
        return {"output": [], "errors": errors}
```

## Configuration (Environment Variables)

| Variable                     | Default           | Options                             |
| ---------------------------- | ----------------- | ----------------------------------- |
| `AUTO_ANALYST_LOG_LEVEL`     | `DEBUG`           | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AUTO_ANALYST_LOG_FORMAT`    | `plain`           | `plain`, `json`                     |
| `AUTO_ANALYST_LOG_FILE`      | `auto_analyst.log`| Path to log file                    |
| `AUTO_ANALYST_LOG_REDACT_QUERIES` | `false`      | `true`, `false`                     |

## Output Formats

**Plain (default):**

```
2025-01-15 10:30:45 INFO [api.graph.plan] [run=abc-123] plan_complete
```

**JSON (`AUTO_ANALYST_LOG_FORMAT=json`):**

```json
{
  "time": "2025-01-15 10:30:45",
  "level": "INFO",
  "name": "api.graph.plan",
  "message": "plan_complete",
  "run_id": "abc-123"
}
```

## Log Formatters

The logging system uses custom formatters in `api/logging_setup.py`:

```python
class _DefaultFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        return super().format(record)

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)
```

## Query Redaction

For privacy, enable query redaction in production:

```bash
AUTO_ANALYST_LOG_REDACT_QUERIES=true
```

This redacts sensitive fields (`query`, `task_queries`, `urls`) in log output.

## Best Practices

1. **Use structured extras** for metrics:

   ```python
   log.info("search_complete", extra={"results": len(results), "duration_ms": elapsed})
   ```

2. **Use `log.exception()` for errors** (auto-captures stack trace):

   ```python
   except Exception:
       log.exception("operation_failed")
   ```

3. **Logger names follow module paths**:

   - `api.graph.plan`, `api.graph.search`, etc. for pipeline nodes
   - `__name__` for other modules

4. **Pass `run_id` through function calls** when logging outside nodes:

   ```python
   def fetch_url(result: SearchResult, run_id: str | None = None):
       log = get_logger("tools.fetcher", run_id=run_id)
   ```

5. **Log start/complete events** with timing:

   ```python
   log.info("operation_start", extra={"input_count": len(data)})
   # ... work ...
   log.info("operation_complete", extra={
       "output_count": len(result),
       "duration_ms": (perf_counter() - start) * 1000
   })
   ```

6. **Include relevant context** in log extras:

   ```python
   log.info("fetch_complete", extra={
       "documents": len(documents),
       "from_results": len(search_results),
       "failed": fetch_failed,
       "chunks": len(chunks),
   })
   ```

## Standard Log Event Names

Use consistent event names across the codebase:

| Event Pattern          | Usage                                    |
| ---------------------- | ---------------------------------------- |
| `*_start`              | Beginning of an operation                |
| `*_complete`           | Successful completion                    |
| `*_failed`             | Operation failure                        |
| `*_skip`               | Operation skipped (with reason)          |
| `*_retry`              | Retrying an operation                    |
| `*_cache_hit`          | Retrieved from cache                     |
| `*_cache_miss`         | Not found in cache                       |
| `*_rate_limited`       | API rate limit encountered               |

## File Locations

- `api/logging_setup.py` — Logger factory and formatters
- `api/config.py` — Log configuration constants

## Production Recommendations

- Default to `AUTO_ANALYST_LOG_FILE=auto_analyst.log` for file output
- Use JSON format in production (`AUTO_ANALYST_LOG_FORMAT=json`)
- Use INFO level for normal operation, DEBUG for troubleshooting
- Ensure `run_id` is present in all log lines
- Enable query redaction for sensitive deployments
- Do not disable logging in tests; accept file output in repo root
