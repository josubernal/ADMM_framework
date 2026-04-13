## Installation

To install the required dependencies for this project, you can use `uv`. If you don't have `uv` installed, please follow the instructions on the [official uv website](https://github.com/astral-sh/uv).

Once `uv` is installed, navigate to the project's root directory (where `pyproject.toml` is located) in your terminal and run the following command:

```bash
uv sync --no-dev
```

This command will create a virtual environment (if one doesn't exist) and install all the dependencies specified in `pyproject.toml`.

If you want to run test also to verify math is correct and check bugs, intead run:
```bash
uv sync 
```
To run the tests
```bash
uv run pytest -s admm/tests/
```