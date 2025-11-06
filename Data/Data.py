import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Data:
    
    def __init__(self, dataset_id=1494, test_size=0.2, random_state=42):
        self.dataset_id = dataset_id
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.data = None
        self.feature_names = None
        self.target_name = None  


    def load_data(self):
        # Cargar dataset desde OpenML
        dataset = openml.datasets.get_dataset(self.dataset_id, download_all_files=True)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        self.data = pd.concat([X, y], axis=1)
        self.feature_names = X.columns.tolist()
        self.target_name = y.name
        print("Dataset cargado con éxito.")
        print(f"Número de columnas: {self.data.shape[1]}")
        print(f"Tipos de datos:\n{self.data.dtypes}")
        print(f"Valores únicos de la variable objetivo ({self.target_name}): {y.unique()}")

    def preprocess(self):
        # Eliminar filas con NaN en la variable objetivo
        self.data = self.data.dropna(subset=[self.target_name])
            
        # Convertir la variable objetivo a 0/1
        self.data[self.target_name] = self.data[self.target_name].map({'1': 0, '2': 1})

        # Dividir en entrenamiento y prueba
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        # Escalar los datos
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Preprocesamiento completado: variables objetivo convertidas, datos divididos y escalados.")


    #dataset completo como artifact
    def save_as_artifact(self, path="dataset"):
        """
        Guarda el dataset completo (X e y) como un artefacto CSV.
        """
        if self.data is not None:
            # Crear un DataFrame combinado para guardar
            combined_df = pd.DataFrame(self.X_train, columns=self.feature_names)
            combined_df[self.target_name] = self.y_train
            
            # Guardar el DataFrame combinado en un archivo CSV
            filepath = f"{path}.csv"
            combined_df.to_csv(filepath, index=False)
            return filepath
        return None
        
