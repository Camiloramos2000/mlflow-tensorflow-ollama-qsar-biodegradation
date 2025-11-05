import pandas as pd
from rich.console import Console
import uuid

console = Console()

def get_last_run(mlflow, client):
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

    # ------------------------------
    # Comparación de métricas de ambos modelos
    # ------------------------------


    if logistic_metrics and nn_metrics:
        all_metrics_keys = sorted(set(logistic_metrics.keys()).union(nn_metrics.keys()))
        comparison_df = pd.DataFrame({
            "Logistic_Regression": [logistic_metrics.get(k, None) for k in all_metrics_keys],
            "Neural_Network": [nn_metrics.get(k, None) for k in all_metrics_keys]
        }, index=all_metrics_keys)

        # Guardar CSV local
        comparison_file = "metrics_comparison.csv"
        comparison_df.to_csv(comparison_file)
        console.print(f"📊 Comparación de métricas generada:\n{comparison_df}", style="green")

        # Registrar en MLflow
        mlflow.set_experiment("Metrics_Comparison")
        with mlflow.start_run(run_name=f"Metrics_Comparison_{uuid.uuid4()}", nested=True):
            mlflow.log_artifact(comparison_file)
            console.print(f"✅ Comparación de métricas registrada como artefacto en MLflow: {comparison_file}", style="green bold")
    else:
        console.print("⚠️ No se pudieron obtener métricas de ambos modelos para la comparación.", style="yellow")