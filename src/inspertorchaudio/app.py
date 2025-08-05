import typer

app = typer.Typer()

@app.command('hello')
def hello(name: str = "World"):
    typer.echo(f"Hello {name}!")

@app.command('fma-demo')
def fma_demo(
    experiment_name: str = 'baseline_fma',
    description: str = 'Baseline FMA small dataset with Dieleman2014 model',
    model_name: str = 'Dieleman2014',
):
    from inspertorchaudio.experiments.supervised_train_test import fma_small_demo
    fma_small_demo(
        experiment_name=experiment_name,
        description=description,
        model_name=model_name,
    )
    
if __name__ == "__main__":
    app()