# AGENTS.md

Minimal agent index for working in Auto-Analyst.

## Canonical Docs

| Topic | Source |
| --- | --- |
| Project overview, setup, runtime, architecture | `README.md` |
| Canonical command surface | `Makefile` |
| CI checks and environment | `.github/workflows/ci.yml` |
| Architecture boundaries | `docs/instructions/clean-architecture.md` |
| State and pipeline patterns | `docs/instructions/state-management.md` |
| Logging contract | `docs/instructions/logging.md` |
| LLM integration contract | `docs/instructions/llm-integration.md` |
| Vector store patterns | `docs/instructions/vector-store.md` |
| Testing patterns | `docs/instructions/testing.md` |
| Evaluation patterns | `docs/instructions/evaluation.md` |

## Golden Commands

```bash
make setup
make run
make run-chainlit
make lint
make test
make check
make build
make format
make eval
```

## Local Validation

```bash
python -m venv .venv
source .venv/bin/activate
make setup
make lint
make test
```

## Conventions (Short)

- Keep layer boundaries: `api/` and `tools/` must not import from `ui/`.
- Use `get_logger()` from `api/logging_setup.py`, not `logging.getLogger()`.
- Pipeline nodes should return only modified state keys and preserve `errors`/`warnings`.
- Mock external I/O in tests; keep tests deterministic.
- Use existing tooling only for guardrails (`py_compile` + `pytest` in CI).

## Guardrail Limitations

- No repo-pinned formatter/type-check framework is currently enforced.
- `make format` is informational until formatter tooling is standardized in-repo.
