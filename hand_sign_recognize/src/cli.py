"""Console script for hand_sign_recognize."""
import hand_sign_recognize as model

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for hand_sign_recognize."""
    console.print("Replace this message by putting your code into "
               "hand_sign_recognize.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    model_path="../Scripts/best_model_lenet.h5"
    model.run(model_path)



if __name__ == "__main__":
    app()
