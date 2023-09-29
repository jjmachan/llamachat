import os
import sys
from pathlib import Path

import click
from streamlit.web import cli as stcli


@click.command()
@click.argument("path")
def cli(path: str):
    """
    Run LlamaChat in CLI mode
    """
    # set storage path
    os.environ["LLAMA_INDEX_PATH"] = path

    app_path = Path(__file__).parent / "app.py"
    print("running streamlit...")
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    cli()
