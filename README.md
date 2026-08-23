# ncert-rag

A complete RAG pipeline for NCERT books

## Use it

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14+. `uv sync` fetches the
interpreter if you don't already have one.

| Command | |
| --- | --- |
| `uv sync` | create `.venv` and install dependencies |
| `uv run ncert-rag` | run the CLI |
| `uv run pytest` | run the tests |
| `uv run ruff check --fix .` | lint |
| `uv run ruff format .` | format |
| `uvx pre-commit install` | lint and format on every commit, once per clone |

### Companion tool

The textbooks are fetched by [ncert-cli](https://github.com/amanverma-765/ncert-cli)
(`pip install ncert-cli`), a downloader built for this pipeline. `uv sync` installs it;
`main()` fetches Class X Mathematics and Science into `data/` on first run.

### Layout

- `src/ncert_rag/main.py` — `main()`, what the console script calls
- `src/ncert_rag/download.py` — fetches the textbooks via ncert-cli
- `src/ncert_rag/parser.py` — samples pages and counts tokens
- `tests/test_main.py` — its test
- `pyproject.toml` — dependencies, entry point, ruff and pytest config

### Dependencies

`uv add <pkg>` for runtime, `uv add --dev <pkg>` for tooling. Both write to
`uv.lock`, which is committed so every clone resolves to the same versions.
