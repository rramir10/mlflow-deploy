import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature
import sys
import traceback
import joblib  # <--- AGREGAR ESTE IMPORT

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

# --- Asegurar que el directorio MLRuns exista ---
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o Establecer Experimento ---
experiment_name = "CI-CD-Lab2"
experiment_id = None

try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. Obteniendo ID. ---")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
            print(f"--- Debug: Ubicación de Artefacto del Experimento Existente: {experiment.artifact_location} ---")
            if experiment.artifact_location != artifact_location:
                print(f"--- WARNING: La ubicación del artefacto del experimento existente ('{experiment.artifact_location}') NO coincide con la deseada ('{artifact_location}')! ---")
        else:
            print(f"--- ERROR: No se pudo obtener el experimento existente '{experiment_name}' por nombre. ---")
            sys.exit(1)
    else:
        print(f"--- ERROR creando/obteniendo experimento: {e} ---")
        raise e

if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido para '{experiment_name}'. ---")
    sys.exit(1)

# --- Cargar Datos y Entrenar Modelo ---
print("--- Debug: Cargando datos de diabetes ---")
X, y = load_diabetes(return_X_y=True)

# <--- AGREGAR random_state para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Debug: Entrenando modelo ---")
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print(f"--- Debug: MSE calculado: {mse:.4f} ---")

# --- Iniciar Run de MLflow ---
print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---")
run = None

try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")

        expected_artifact_uri_base = os.path.join(artifact_location, run_id, "artifacts")
        if actual_artifact_uri != expected_artifact_uri_base:
            print(f"--- WARNING: La URI del Artefacto del Run '{actual_artifact_uri}' no coincide exactamente con la esperada '{expected_artifact_uri_base}' ---")
        
        if "/home/manuelcastiblan/" in actual_artifact_uri:
            print(f"--- ¡¡¡ERROR CRÍTICO!!!: La URI del Artefacto del Run '{actual_artifact_uri}' TODAVÍA contiene la ruta local incorrecta! ---")

        # Registrar métricas en MLflow
        mlflow.log_metric("mse", mse)
        print(f"--- Debug: Intentando log_model con artifact_path='model' ---")

        # Registrar modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        print(f"✅ Modelo registrado en MLflow. MSE: {mse:.4f}")

    # ============================================================================
    # 🔴 AGREGAR ESTA SECCIÓN - GUARDAR MODELO COMO .PKL
    # ============================================================================
    
    print("\n--- Debug: Guardando modelo como .pkl para validación ---")
    model_path = "model.pkl"
    
    try:
        joblib.dump(model, model_path)
        print(f"✅ Modelo guardado exitosamente en: {os.path.abspath(model_path)}")
        
        # Verificar que se creó
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"--- Debug: Tamaño del archivo: {file_size} bytes ---")
        else:
            print(f"❌ ERROR: El archivo {model_path} no se creó correctamente")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ ERROR guardando modelo como .pkl: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ============================================================================
    
    print("\n" + "="*70)
    print("✅ PIPELINE DE ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"📊 MSE: {mse:.4f}")
    print(f"📁 Modelo guardado en: {model_path}")
    print(f"📦 MLflow run ID: {run_id}")
    print("="*70)

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- Fin de la Traza de Error ---")
    print(f"CWD actual en el error: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID intentado: {experiment_id}")
    if run:
        print(f"URI del Artefacto del Run en el error: {run.info.artifact_uri}")
    else:
        print("El objeto Run no se creó con éxito.")
    sys.exit(1)