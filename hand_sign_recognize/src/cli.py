"""Console script for hand_sign_recognize."""
from predict import predict_by_video


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
    # type:
    # "video" (bật camera và dự đoán liên tục)
    # "photo" (bật camera nhấn "c" để chụp ảnh và dự đoán)
    predict_by_video(model_path,"photo")

if __name__ == "__main__":
    app()
