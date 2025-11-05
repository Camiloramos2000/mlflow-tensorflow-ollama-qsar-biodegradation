import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from rich.console import Console
from rich.panel import Panel

console = Console()

class NeuralNetworkModel:
    def __init__(self, input_dim=None, layers=None, activations=None,
                 output_activation='sigmoid', optimizer='adam', loss='binary_crossentropy',
                 metrics=None, epochs=50, batch_size=32):
        """
        Inicializa el modelo secuencial de Keras.
        Parámetros epochs y batch_size ahora pueden pasarse al constructor.
        """
        # evitar listas mutables como valores por defecto
        self.input_dim = input_dim
        self.layers = layers if layers is not None else [128, 64]
        self.activations = activations if activations is not None else ['relu', 'relu']
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics if metrics is not None else ['accuracy']
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.name = "neural_network"

    # ======================================================
    # MÉTODO PARA CONFIGURAR PARÁMETROS DESDE CONSOLA
    # ======================================================
    def set_params(self, data_loader):
        console.print(Panel.fit("✏️ Parámetros para la Red Neuronal", style="cyan"))

        # ==============================
        # Input dimension
        # ==============================
        self.input_dim = data_loader.X_train.shape[1]
        console.print(f"input_dim (auto): [white]{self.input_dim}[/white]", style="white")

        # ==============================
        # Hidden layers
        # ==============================
        console.print(
            "\n🧠 [blue]Capas ocultas[/blue]\n"
            "Define cuántas capas tendrá tu red y cuántas neuronas por capa.\n"
            "👉 Ejemplo: '128,64' crea dos capas ocultas, una con 128 neuronas y otra con 64.\n"
            "💡 [italic]Tip:[/italic] No uses demasiadas capas si el dataset es pequeño; puede sobreajustar.\n",
            style="blue"
        )
        while True:
            layers_input = console.input("[yellow]Capas ocultas (ej. '128,64'):[/yellow]\n[white]> [/white]")
            try:
                layers = [int(x.strip()) for x in layers_input.split(',') if x.strip()]
                if all(l > 0 for l in layers):
                    console.print(f"✅ Capas aceptadas: {layers}\n", style="green bold")
                    break
                else:
                    console.print("❌ Todas las capas deben tener un número positivo de neuronas.\n", style="red bold")
            except ValueError:
                console.print("❌ Entrada inválida. Usa solo números separados por comas (ej. 128,64).\n", style="red bold")
        self.layers = layers

        # ==============================
        # Activation functions
        # ==============================
        console.print(
            "\n⚡ [blue]Funciones de activación[/blue]\n"
            "Estas funciones determinan cómo se activan las neuronas.\n"
            "👉 Ejemplo: 'relu,relu' corresponde a las mismas capas que definiste.\n"
            "Opciones comunes: [yellow]relu[/yellow], [yellow]tanh[/yellow], [yellow]sigmoid[/yellow].\n"
            "💡 [italic]Tip:[/italic] Usa 'relu' para capas ocultas y 'sigmoid' o 'softmax' para salida.\n",
            style="blue"
        )
        while True:
            activations_input = console.input("[yellow]Funciones de activación (ej. 'relu,relu'):[/yellow]\n[white]> [/white]")
            activations = [x.strip().lower() for x in activations_input.split(',') if x.strip()]
            if len(activations) == len(self.layers):
                console.print(f"✅ Funciones de activación aceptadas: {activations}\n", style="green bold")
                break
            else:
                console.print(f"❌ Debes ingresar exactamente {len(self.layers)} funciones, una por capa.\n", style="red bold")
        self.activations = activations

        # ==============================
        # Output activation
        # ==============================
        console.print(
            "\n🎯 [blue]Función de activación de salida[/blue]\n"
            "Controla cómo se interpreta la salida del modelo.\n"
            "👉 Para clasificación binaria, usa [yellow]'sigmoid'[/yellow].\n"
            "👉 Para multiclase, usa [yellow]'softmax'[/yellow].\n",
            style="blue"
        )
        while True:
            output_activation = console.input("[yellow]Función de activación de salida ('sigmoid' o 'softmax'):[/yellow]\n[white]> [/white]").strip().lower()
            if output_activation in ["sigmoid", "softmax"]:
                console.print(f"✅ Función de salida: {output_activation}\n", style="green bold")
                break
            else:
                console.print("❌ Valor inválido. Usa 'sigmoid' o 'softmax'.\n", style="red bold")
        self.output_activation = output_activation

        # ==============================
        # Optimizer
        # ==============================
        console.print(
            "\n⚙️ [blue]Optimizador[/blue]\n"
            "Define cómo se actualizan los pesos del modelo en cada paso de entrenamiento.\n"
            "Opciones comunes: [yellow]adam[/yellow], [yellow]rmsprop[/yellow], [yellow]sgd[/yellow].\n"
            "💡 [italic]Tip:[/italic] 'adam' suele funcionar muy bien en la mayoría de los casos.\n",
            style="blue"
        )
        valid_opts = ["adam", "rmsprop", "sgd"]
        while True:
            optimizer = console.input("[yellow]Optimizador ('adam', 'rmsprop', 'sgd'):[/yellow]\n[white]> [/white]").strip().lower()
            if optimizer in valid_opts:
                console.print(f"✅ Optimizador aceptado: {optimizer}\n", style="green bold")
                break
            else:
                console.print(f"❌ Optimizador no válido. Opciones: {', '.join(valid_opts)}.\n", style="red bold")
        self.optimizer = optimizer

        # ==============================
        # Loss function
        # ==============================
        console.print(
            "\n💔 [blue]Función de pérdida[/blue]\n"
            "Mide qué tan bien está aprendiendo el modelo.\n"
            "👉 Para clasificación binaria: [yellow]'binary_crossentropy'[/yellow].\n"
            "👉 Para multiclase: [yellow]'categorical_crossentropy'[/yellow].\n",
            style="blue"
        )
        valid_losses = ["binary_crossentropy", "categorical_crossentropy"]
        while True:
            loss = console.input("[yellow]Función de pérdida:[/yellow]\n[white]> [/white]").strip().lower()
            if loss in valid_losses:
                console.print(f"✅ Función de pérdida aceptada: {loss}\n", style="green bold")
                break
            else:
                console.print(f"❌ Valor inválido. Usa una de: {', '.join(valid_losses)}.\n", style="red bold")
        self.loss = loss

        # ==============================
        # Epochs
        # ==============================
        console.print(
            "\n📆 [blue]Número de épocas[/blue]\n"
            "Define cuántas veces el modelo verá el conjunto completo de datos durante el entrenamiento.\n"
            "💡 [italic]Tip:[/italic] Empieza con 20 o 50; más épocas pueden mejorar el aprendizaje, pero también sobreajustar.\n",
            style="blue"
        )
        while True:
            try:
                epochs = int(console.input("[yellow]Número de épocas:[/yellow]\n[white]> [/white]"))
                if 1 <= epochs <= 500:
                    console.print(f"✅ Número de épocas aceptado: {epochs}\n", style="green bold")
                    break
                else:
                    console.print("❌ Debe estar entre 1 y 500.\n", style="red bold")
            except ValueError:
                console.print("❌ Ingrese un número entero válido.\n", style="red bold")

        # ==============================
        # Batch size
        # ==============================
        console.print(
            "\n📦 [blue]Tamaño del batch[/blue]\n"
            "Número de muestras que se procesan antes de actualizar los pesos.\n"
            "💡 [italic]Tip:[/italic] 32 o 64 son tamaños comunes; valores más grandes usan más memoria pero entrenan más rápido.\n",
            style="blue"
        )
        while True:
            try:
                batch_size = int(console.input("[yellow]Tamaño del batch:[/yellow]\n[white]> [/white]"))
                if 8 <= batch_size <= 512:
                    console.print(f"✅ Tamaño del batch aceptado: {batch_size}\n", style="green bold")
                    break
                else:
                    console.print("❌ El tamaño del batch debe estar entre 8 y 512.\n", style="red bold")
            except ValueError:
                console.print("❌ Ingrese un número entero válido.\n", style="red bold")

        # asignar a la instancia
        self.epochs = epochs
        self.batch_size = batch_size

        console.print("\n✅ [green bold]Parámetros de red neuronal configurados correctamente.[/green bold]\n")
        return {
            "input_dim": self.input_dim,
            "layers": self.layers,
            "activations": self.activations,
            "output_activation": self.output_activation,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

    # ======================================================
    # CONSTRUCCIÓN DEL MODELO
    # ======================================================
    def build(self):
        model = Sequential()
        for i, (neurons, activation) in enumerate(zip(self.layers, self.activations)):
            if i == 0:
                model.add(Dense(neurons, activation=activation, input_dim=self.input_dim))
            else:
                model.add(Dense(neurons, activation=activation))
        model.add(Dense(1, activation=self.output_activation))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model = model
        console.print("✅ [green]Red neuronal construida y compilada correctamente.[/green]")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            console.print("⚙️ [yellow]Construyendo modelo antes de entrenar...[/yellow]")
            self.build()

        console.print(f"🚀 Entrenando red neuronal durante [cyan]{self.epochs}[/cyan] épocas con batch [cyan]{self.batch_size}[/cyan]...\n")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2
        )

        console.print("\n✅ [green bold]Entrenamiento finalizado correctamente.[/green bold]")
        return history
