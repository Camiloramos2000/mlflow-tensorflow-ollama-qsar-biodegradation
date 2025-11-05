import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from rich.console import Console

console = Console()

class LogisticModel:
    """
    Modelo de regresión logística para clasificación binaria.
    """

    def __init__(self, C=1.0, max_iter=100, solver='lbfgs'):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
        self.name = "logistic_regression_model"
        self.metrics = None
        self.conf_matrix_path = None

    def train(self, X_train, y_train):
        """Entrena el modelo con los datos de entrenamiento."""
        self.model.fit(X_train, y_train)
        console.print("✅ Modelo entrenado con éxito.\n", style="green bold")

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo y guarda la matriz de confusión."""
        y_pred = self.model.predict(X_test)
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(self.model, X_test, y_test, ax=ax)
        self.conf_matrix_path = "confusion_matrix.png"
        plt.savefig(self.conf_matrix_path)
        plt.close(fig)
        console.print("📊 Matriz de confusión guardada en 'confusion_matrix.png'\n", style="blue bold")
        return self.metrics

    def set_params(self):
        """Permite configurar los hiperparámetros con validaciones visuales."""
        
        # === Funciones de validación ===
        def get_valid_C():
            while True:
                try:
                    console.print(
                        "\n🧠 [yellow]C[/yellow] controla la fuerza de la regularización del modelo de regresión logística.\n"
                        "👉 Valores pequeños (por ejemplo 0.1) aplican una regularización fuerte: el modelo se simplifica y evita sobreajuste.\n"
                        "👉 Valores grandes (por ejemplo 10.0) reducen la regularización: el modelo se ajusta más a los datos, pero puede sobreajustar.\n"
                        "💡 Recomendado: empieza con un valor medio como 1.0 y ajusta según el rendimiento.\n",
                        style="blue"
                    )
                    console.print("Ingrese el valor de [yellow]C[/yellow] (rango 0.1 - 10.0):", style="yellow")

                    value = float(console.input("[white]> [/white]"))
                    if 0.1 <= value <= 10.0:
                        console.print(f"✅ Valor aceptado: C = {value}\n", style="green bold")
                        return value
                    else:
                        console.print("❌ Error: El valor de C debe estar entre 0.1 y 10.0.\n", style="red bold")
                except ValueError:
                    console.print("❌ Entrada inválida. Ingrese un número decimal válido.\n", style="red bold")

        def get_valid_max_iter():
            while True:
                try:
                    console.print(
                        "\n🧠 [blue]¿Qué es el parámetro max_iter?[/blue]\n"
                        "Este parámetro indica el [bold]número máximo de iteraciones[/bold] que el algoritmo de optimización realizará "
                        "para ajustar los coeficientes del modelo de regresión logística.\n\n"
                        "👉 Si el modelo tarda en converger (no logra estabilizar los pesos), puedes [yellow]aumentar[/yellow] este valor.\n"
                        "👉 Si el modelo converge muy rápido, puedes [yellow]reducir[/yellow]lo para ahorrar tiempo de entrenamiento.\n\n"
                        "💡 [italic]Tip:[/italic] Un valor típico es entre [bold]300[/bold] y [bold]500[/bold]. "
                        "Si recibes advertencias de 'no converge', aumenta hasta 1000.\n",
                        style="blue"
                    )

                    console.print("Ingrese el valor de [yellow]max_iter[/yellow] (rango 100 - 1000):", style="yellow")


                    value = int(console.input("[white]> [/white]"))
                    if 100 <= value <= 1000:
                        console.print(f"✅ Valor aceptado: max_iter = {value}\n", style="green bold")
                        return value
                    else:
                        console.print("❌ Error: max_iter debe estar entre 100 y 1000.\n", style="red bold")
                except ValueError:
                    console.print("❌ Entrada inválida. Ingrese un número entero válido.\n", style="red bold")

        def get_valid_solver():
            valid_solvers = ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]
            while True:
                console.print(
                    "\n🧠 [blue]¿Qué es el parámetro solver?[/blue]\n"
                    "El parámetro [bold]solver[/bold] define el [bold]algoritmo de optimización[/bold] que usa la regresión logística "
                    "para encontrar los coeficientes del modelo.\n\n"
                    "Cada solver tiene características diferentes y puede funcionar mejor según el tamaño del dataset o el tipo de regularización:\n"
                    "• [yellow]liblinear[/yellow]: recomendado para datasets pequeños; solo soporta regularización L1 y L2.\n"
                    "• [yellow]lbfgs[/yellow]: rápido y eficiente, funciona bien en la mayoría de los casos (default).\n"
                    "• [yellow]newton-cg[/yellow]: similar a lbfgs, útil para problemas grandes.\n"
                    "• [yellow]sag[/yellow] y [yellow]saga[/yellow]: buenos para datasets muy grandes.\n\n"
                    "💡 [italic]Tip:[/italic] Si no estás seguro, usa [bold]'lbfgs'[/bold] (es estable y preciso en la mayoría de los casos).\n",
                    style="blue"
                )

                console.print("Ingrese el [yellow]solver[/yellow] ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'):", style="yellow")


                value = console.input("[white]> [/white]").strip().lower()
                if value in valid_solvers:
                    console.print(f"✅ Solver aceptado: {value}\n", style="green bold")
                    return value
                else:
                    console.print(f"❌ Solver no válido. Opciones permitidas: {', '.join(valid_solvers)}.\n", style="red bold")

        # === Interacción con el usuario ===
        console.print("\n⚙️  Configuración de hiperparámetros para [blue]Regresión Logística[/blue]\n", style="blue bold")

        self.C = get_valid_C()
        self.max_iter = get_valid_max_iter()
        self.solver = get_valid_solver()

        # === Actualizar modelo ===
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=42
        )

        console.print("🔁 Modelo actualizado con los nuevos hiperparámetros.\n", style="green bold")

        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": self.solver
        }
