# core_mlflow_console.py

# ==============================
# Imports
# ==============================
import os
import uuid
import time
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.keras
import tensorflow as tf
from mlflow.models.signature import infer_signature
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt

from Data.Data import Data
from models.Logisitc_Regression import LogisticModel
from models.NeuralNetworkModel import NeuralNetworkModel
from models.Ollama import OllamaInterpreter
from models.Ollama import OllamaPyfunc
from get_runs import get_last_run



# ==============================
# Configuración Global
# ==============================
console = Console()


# ==============================
# Funciones auxiliares
# ==============================
def animated_print(text, delay=0.02, style="white"):
    """Imprime texto con efecto de máquina de escribir"""
    for char in text:
        console.print(char, end="", style=style, highlight=False)
        time.sleep(delay)
    console.print()


def clear_console():
    """Limpia la consola"""
    os.system("cls" if os.name == "nt" else "clear")


def prompt_input(message, prompt_type="str"):
    """Muestra inputs con colores consistentes"""
    console.print(message, style="yellow")
    if prompt_type == "float":
        return FloatPrompt.ask("", default=None, show_default=False)
    elif prompt_type == "int":
        return IntPrompt.ask("", default=None, show_default=False)
    else:
        return Prompt.ask("", default=None, show_default=False)


# ==============================
# Ejecución principal
# ==============================
if __name__ == "__main__":
    client = mlflow.tracking.MlflowClient()

    console.rule("[bold cyan]📝 Evaluación de Modelos[/bold cyan]")

    console.print("1️⃣  Regresión Logística", style="magenta")
    console.print("2️⃣  Red Neuronal", style="magenta")

    # ------------------------------
    # Cargar dataset
    # ------------------------------
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="[cyan]📂 Cargando dataset...", total=None)
        try:
            data_loader = Data(dataset_id=1494)
            data_loader.load_data()
            data_loader.preprocess()
            time.sleep(0.5)
            console.print("✅ Dataset cargado y preprocesado.", style="green bold")
        except Exception as e:
            console.print(f"❌ Error al cargar el dataset: {e}", style="red bold")
            exit(1)

    # ------------------------------
    # Entrenamiento Regresión Logística
    # ------------------------------
    console.print(Panel.fit("✏️ Parámetros para la Regresión Logística", style="cyan"))

    logistic_model = LogisticModel()
    hyperparams = logistic_model.set_params()

    C = hyperparams["C"]
    max_iter = hyperparams["max_iter"]
    solver = hyperparams["solver"]

    mlflow.set_experiment("Logistic_Regression_Model")

    with mlflow.start_run(run_name=f"Logistic_Regression_Training_{uuid.uuid4()}") as run:
        data_loader.save_as_artifact()

        try:
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("solver", solver)

            animated_print("\n⚙️  Entrenando Regresión Logística...", style="yellow")
            logistic_model.train(data_loader.X_train, data_loader.y_train)

            animated_print("📊 Evaluando modelo...", style="yellow")
            logistic_metrics = logistic_model.evaluate(data_loader.X_test, data_loader.y_test)
            mlflow.log_metrics(logistic_metrics)

            animated_print("📦 Registrando artefactos y modelo en MLflow...", style="yellow")
            if logistic_model.conf_matrix_path:
                mlflow.log_artifact(logistic_model.conf_matrix_path)

            signature = infer_signature(data_loader.X_train, logistic_model.model.predict(data_loader.X_train))
            model_info = mlflow.sklearn.log_model(
                sk_model=logistic_model.model,
                name=logistic_model.name,
                signature=signature
            )

            # Registrar modelo
            
            try:
                client.create_registered_model(
                    name=logistic_model.name,
                    description="Command classification using Logistic Regression"
                )
                console.print("✅ Modelo registrado en MLflow Model Registry.", style="green bold")
            except mlflow.exceptions.MlflowException as e:
                if "already exists" in str(e):
                    console.print(f"⚠️ Modelo '{logistic_model.name}' ya existe. Usando el existente.", style="yellow")

            console.rule("[blue]🆕 Creando nueva versión del modelo[/blue]")
            mv = client.create_model_version(
                name=logistic_model.name,
                source=model_info.model_uri,
                run_id=run.info.run_id
            )
            client.update_model_version(
                name=mv.name,
                version=mv.version,
                description="Logistic Regression"
            )
            console.print("✅ Versión del modelo creada/actualizada.\n", style="green bold")

        except Exception as e:
            console.print(f"❌ Error durante el entrenamiento: {e}", style="red bold")

    # ------------------------------
    # Entrenamiento Red Neuronal
    # ------------------------------
    console.print(Panel.fit("✏️ Parámetros para la Red Neuronal", style="cyan"))

    nn_model = NeuralNetworkModel()
    hyperparams = nn_model.set_params(data_loader)

    input_dim = hyperparams["input_dim"]
    layers = hyperparams["layers"]
    activations = hyperparams["activations"]
    output_activation = hyperparams["output_activation"]
    optimizer = hyperparams["optimizer"]
    loss = hyperparams["loss"]
    epochs = hyperparams["epochs"]
    batch_size = hyperparams["batch_size"]

    mlflow.set_experiment("Neural_Network_Model")

    with mlflow.start_run(run_name=f"Neural_Network_Training_{uuid.uuid4()}") as run:
        #guardar dataset como artifact:
        data_loader.save_as_artifact()
        
        try:
            mlflow.tensorflow.autolog()

            mlflow.log_param("input_dim", input_dim)
            mlflow.log_param("layers", str(layers))
            mlflow.log_param("activations", str(activations))
            mlflow.log_param("output_activation", output_activation)
            mlflow.log_param("optimizer", optimizer)
            mlflow.log_param("loss", loss)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            metrics_objects = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]

            animated_print("\n⚙️  Construyendo y entrenando la Red Neuronal...", style="yellow")

            nn_model = NeuralNetworkModel(
                input_dim=input_dim,
                layers=layers,
                activations=activations,
                output_activation=output_activation,
                optimizer=optimizer,
                loss=loss,
                metrics=metrics_objects,
                epochs=epochs,
                batch_size=batch_size
            )
            
            nn_model.train(
                data_loader.X_train, data_loader.y_train,
                X_val=data_loader.X_test, y_val=data_loader.y_test
            )

            animated_print("📦 Registrando modelo en MLflow...", style="yellow")
            signature = infer_signature(data_loader.X_train, nn_model.model.predict(data_loader.X_train))
            
            mlflow.tensorflow.log_model(
                model=nn_model.model,
                name=nn_model.name,
                registered_model_name=nn_model.name,
                signature=signature,
                pip_requirements=["tensorflow==2.14.1", "cloudpickle==3.1.2"]
            )

            console.print("✅ Modelo de red neuronal registrado correctamente.", style="green bold")

        except Exception as e:
            console.print(f"❌ Error en el entrenamiento de la red neuronal: {e}", style="red bold")

    #Ollama interpreta los ultimos resultados guardados en MLFLOW CON   Usa mlflow.search_runs()
    

    get_last_run(mlflow, client)
    
    
    # ------------------------------
    # Ollama interpreta los ultimos resultados guardados en MLFLOW
    # ------------------------------
    console.print(Panel.fit("🤖 Interpretación de Resultados con Ollama", style="cyan"))

    ollama_interpreter = OllamaInterpreter()

    if not ollama_interpreter.check_ollama():
        console.print("🚫 [red]Ollama no está disponible. Saltando la interpretación con IA.[/red]")
    else:
        console.print("✨ [green]Ollama está listo para interpretar los resultados.[/green]\n")

        # Obtener el último run de Regresión Logística
        logistic_runs = mlflow.search_runs(
            experiment_names=["Logistic_Regression_Model"],
            order_by=["start_time DESC"],
            max_results=1
        )
        logistic_metrics_str = "No se encontraron métricas para Regresión Logística."
        if not logistic_runs.empty:
            logistic_run_id = logistic_runs.iloc[0].run_id
            logistic_metrics = client.get_run(logistic_run_id).data.metrics
            logistic_metrics_str = "\n".join([f"- {k}: {v:.4f}" for k, v in logistic_metrics.items()])
            console.print(f"📊 Últimas métricas de Regresión Logística (Run ID: {logistic_run_id}):\n{logistic_metrics_str}", style="blue")

        # Obtener el último run de Red Neuronal
        nn_runs = mlflow.search_runs(
            experiment_names=["Neural_Network_Model"],
            order_by=["start_time DESC"],
            max_results=1
        )
        nn_metrics_str = "No se encontraron métricas para Red Neuronal."
        if not nn_runs.empty:
            nn_run_id = nn_runs.iloc[0].run_id
            nn_metrics = client.get_run(nn_run_id).data.metrics
            nn_metrics_str = "\n".join([f"- {k}: {v:.4f}" for k, v in nn_metrics.items()])
            console.print(f"📊 Últimas métricas de Red Neuronal (Run ID: {nn_run_id}):\n{nn_metrics_str}", style="blue")
    
    
        questions = [
            f"Analiza y compara ordenamenete, se conciso con los siguientes resultados de cada modelo. "
            f"El primer modelo es una Regresión Logística con las siguientes métricas:\n{logistic_metrics_str}\n"
            f"El segundo modelo es una Red Neuronal con las siguientes métricas:\n{nn_metrics_str}\n"
            f"¿Cuál modelo consideras mejor y por qué? ¿Qué implicaciones tienen estas métricas para la selección del modelo en un contexto de clasificación binaria?",
            f"Basado en las métricas proporcionadas, ¿qué ajustes o mejoras sugerirías para optimizar el rendimiento de la Regresión Logística y la Red Neuronal? "
            f"Considera posibles problemas como el sobreajuste o subajuste, y cómo las métricas (precisión, recall, F1-score, AUC) pueden guiar estas mejoras."
        ]

        time_start = time.time()

        answer = ollama_interpreter.ask_questions(questions)
        len_answer = len(answer)
        time_end = time.time()
        time_elapsed = time_end - time_start
        
        

    console.rule("[bold cyan]Fin del Proceso[/bold cyan]")
    #Guarda las respuestas en un archivo .txt y regístralo en MLflow como artefacto.
    # ------------------------------
    # Registrar respuestas de Ollama en MLflow
    # ------------------------------
    mlflow.set_experiment("Ollama_Interpretation")
    with mlflow.start_run(run_name=f"Ollama_Interpretation_{uuid.uuid4()}", nested=True) as run:
        #guardar datos de los dos modelos que ollama analiza
        mlflow.log_text(logistic_metrics_str, "logistic_regression_metrics.txt")
        mlflow.log_text(nn_metrics_str, "neural_network_metrics.txt")
        
        
        if ollama_interpreter.responses:
            interpretation_file = ollama_interpreter.save_responses()
            if interpretation_file:
                mlflow.log_artifact(interpretation_file)
                console.print(f"✅ Interpretación de Ollama registrada como artefacto en MLflow.", style="green bold")
        else:
            console.print("⚠️ No hay respuestas de Ollama para registrar como artefacto.", style="yellow")
            
        
        
        mlflow.log_param("model_name", "llama2")    
        mlflow.log_metric("ollama_response_count", len_answer)
        mlflow.log_metric("ollama_interpretation_time_seconds", time_elapsed)
        

        
    console.print("\n[bold green]¡Proceso completado![/bold green]")
    console.print("Puedes revisar los resultados en la UI de MLflow ejecutando: [yellow]mlflow ui[/yellow]")
    
        
        
        
        
        
            