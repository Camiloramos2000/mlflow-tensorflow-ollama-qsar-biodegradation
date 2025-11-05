import subprocess
from rich.console import Console
import shutil
import time
import mlflow.pyfunc


console = Console()

class OllamaInterpreter:
    """
    Interactúa con el modelo Ollama (por ejemplo, llama2, llama3) desde consola.
    Incluye validación previa, espera indefinida y salida en tiempo real.
    """

    def __init__(self, model_name="llama2"):
        self.model_name = model_name
        self.responses = []

    def check_ollama(self):
        """
        Verifica que Ollama esté instalado y su servicio esté corriendo.
        """
        # 1. Verificar si el binario existe
        if not shutil.which("ollama"):
            console.print("❌ [red]Ollama no está instalado o no está en el PATH.[/red]")
            console.print("👉 Instálalo desde: https://ollama.ai/download", style="yellow")
            return False

        # 2. Verificar si el servicio de Ollama responde
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                console.print("⚠️ [yellow]Ollama está instalado pero el servicio no responde.[/yellow]")
                console.print("👉 Asegúrate de que el servicio Ollama esté iniciado.", style="yellow")
                return False
        except Exception as e:
            console.print(f"❌ [red]Error al intentar comunicarse con Ollama:[/red] {e}")
            return False

        return True

    def ask_questions(self, questions):
        
        """
        Envía una lista de preguntas al modelo Ollama.
        Espera indefinidamente y muestra la salida en tiempo real.
        """
        self.responses = []

        if not self.check_ollama():
            console.print("🚫 [red]No se puede continuar sin Ollama.[/red]")
            return None

        for q in questions:
            try:
                console.print("\n───────────────────────────────────────────────────────────────", style="cyan")
                console.print(f"🤖 [blue]Ollama responde a:[/blue] [white]{q}[/white]\n")
                
                # Ejecutar el comando con salida en tiempo real
                process = subprocess.Popen(
                    ["ollama", "run", self.model_name, q],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                response = ""
                for line in iter(process.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        console.print(line, style="cyan")
                        response += line + "\n"

                process.wait()

                if process.returncode != 0:
                    error_msg = process.stderr.read().strip()
                    console.print(f"❌ Error al ejecutar Ollama: {error_msg}", style="red bold")
                    self.responses.append((q, f"Error al ejecutar Ollama: {error_msg}"))
                else:
                    console.print("\n✅ [green]Respuesta completa recibida.[/green]\n")
                    self.responses.append(response.strip())

            except KeyboardInterrupt:
                console.print("🛑 Interrumpido manualmente por el usuario.", style="yellow bold")
                self.responses.append((q, "Interrumpido manualmente"))
                break
            except Exception as e:
                console.print(f"❌ Error general con Ollama: {e}", style="red bold")
                self.responses.append((q, f"Error: {e}"))

        return self.responses

    def save_responses(self, filename="ollama_interpretation.txt"):
        """
        Guarda las respuestas generadas por Ollama en un archivo.
        """
        if not self.responses:
            console.print("⚠️ [yellow]No hay respuestas para guardar.[/yellow]")
            return None

        with open(filename, "w", encoding="utf-8") as f:
            for r in self.responses:
                f.write(str(r) + "\n\n")

        console.print(f"📝 Respuestas guardadas en [green]{filename}[/green]")
        return filename


class OllamaPyfunc(mlflow.pyfunc.PythonModel):
    """
    Wrapper para usar OllamaInterpreter desde MLflow.
    """

    def __init__(self, model_name="llama2"):
        self.model_name = model_name

    def load_context(self, context):
        # Cargar la clase OllamaInterpreter
        self.interpreter = OllamaInterpreter(model_name=self.model_name)

    def predict(self, context, model_input):
        """
        model_input: lista de strings (preguntas)
        """
        # Llamar realmente al método ask_questions de Ollama
        if not isinstance(model_input, list):
            model_input = [str(model_input)]
        responses = self.interpreter.ask_questions(model_input)
        return responses
