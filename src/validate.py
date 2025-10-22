"""
Script de Validación de Modelo - VERSIÓN CORREGIDA
===================================================
Esta versión busca el modelo tanto en .pkl como en MLflow
"""

import os
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import traceback

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MSE_THRESHOLD = 5000.0
R2_THRESHOLD = 0.3

print("="*70)
print("🔍 INICIANDO VALIDACIÓN DE MODELO")
print("="*70)
print(f"📏 Umbrales de validación:")
print(f"   • MSE máximo: {MSE_THRESHOLD}")
print(f"   • R² mínimo: {R2_THRESHOLD}")

# =============================================================================
# CARGAR DATOS
# =============================================================================

def load_test_data():
    """Carga los datos de prueba"""
    print("\n📥 Cargando datos de prueba...")
    
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"   ✓ Datos cargados: {X_test.shape[0]} muestras, {X_test.shape[1]} features")
    return X_test, y_test

# =============================================================================
# CARGAR MODELO - MÉTODO MEJORADO
# =============================================================================

def load_trained_model():
    """
    Carga el modelo usando múltiples estrategias:
    1. Desde model.pkl (si existe)
    2. Desde el último run de MLflow
    """
    print("\n🤖 Buscando modelo entrenado...")
    
    # ESTRATEGIA 1: Buscar model.pkl
    model_filename = "model.pkl"
    model_path = os.path.abspath(model_filename)
    
    if os.path.exists(model_path):
        print(f"   ✓ Modelo encontrado en: {model_path}")
        model = joblib.load(model_path)
        print(f"   ✓ Tipo de modelo: {type(model).__name__}")
        return model
    
    # ESTRATEGIA 2: Buscar en MLflow
    print(f"   ⚠️  model.pkl no encontrado, buscando en MLflow...")
    
    try:
        # Configurar MLflow
        workspace_dir = os.getcwd()
        mlruns_dir = os.path.join(workspace_dir, "mlruns")
        tracking_uri = f"file://{os.path.abspath(mlruns_dir)}"
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"   📊 MLflow URI: {tracking_uri}")
        
        # Buscar experimento
        experiment = mlflow.get_experiment_by_name("CI-CD-Lab2")
        
        if not experiment:
            print(f"   ❌ Experimento 'CI-CD-Lab2' no encontrado")
            raise FileNotFoundError("No se encontró el experimento en MLflow")
        
        print(f"   ✓ Experimento encontrado: {experiment.experiment_id}")
        
        # Buscar último run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if len(runs) == 0:
            print(f"   ❌ No se encontraron runs en el experimento")
            raise FileNotFoundError("No hay runs disponibles")
        
        run_id = runs.iloc[0]['run_id']
        print(f"   ✓ Último run encontrado: {run_id}")
        
        # Cargar modelo desde MLflow
        model_uri = f"runs:/{run_id}/model"
        print(f"   📦 Cargando modelo desde: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"   ✓ Modelo cargado desde MLflow")
        print(f"   ✓ Tipo de modelo: {type(model).__name__}")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Error cargando desde MLflow: {e}")
        print(f"\n📁 Directorio actual: {os.getcwd()}")
        print(f"📄 Archivos disponibles:")
        try:
            for item in os.listdir(os.getcwd()):
                print(f"   • {item}")
        except Exception as list_err:
            print(f"   (No se pudo listar: {list_err})")
        
        raise FileNotFoundError(
            "No se pudo cargar el modelo. Verifica que:\n"
            "1. El entrenamiento se ejecutó correctamente\n"
            "2. El archivo model.pkl existe\n"
            "3. O que MLflow tiene el modelo registrado"
        )

# =============================================================================
# EVALUAR MODELO
# =============================================================================

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y calcula métricas"""
    print("\n📊 Evaluando desempeño del modelo...")
    
    try:
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2
        }
        
        print(f"   ✓ MSE: {mse:.4f}")
        print(f"   ✓ RMSE: {rmse:.4f}")
        print(f"   ✓ R² Score: {r2:.4f}")
        
        return metrics
        
    except ValueError as e:
        print(f"\n❌ ERROR durante la predicción: {e}")
        print(f"   • Features en X_test: {X_test.shape[1]}")
        if hasattr(model, 'n_features_in_'):
            print(f"   • Features esperadas por modelo: {model.n_features_in_}")
        raise e

# =============================================================================
# VALIDAR MÉTRICAS
# =============================================================================

def validate_metrics(metrics):
    """Valida si las métricas cumplen los umbrales"""
    print("\n✅ Validando métricas contra umbrales...")
    
    passed = True
    
    # Validar MSE
    print(f"\n   🎯 Validación MSE:")
    print(f"      • Valor actual: {metrics['mse']:.4f}")
    print(f"      • Umbral máximo: {MSE_THRESHOLD}")
    
    if metrics['mse'] <= MSE_THRESHOLD:
        print(f"      ✅ APROBADO (MSE <= {MSE_THRESHOLD})")
    else:
        print(f"      ❌ RECHAZADO (MSE > {MSE_THRESHOLD})")
        passed = False
    
    # Validar R²
    print(f"\n   🎯 Validación R²:")
    print(f"      • Valor actual: {metrics['r2_score']:.4f}")
    print(f"      • Umbral mínimo: {R2_THRESHOLD}")
    
    if metrics['r2_score'] >= R2_THRESHOLD:
        print(f"      ✅ APROBADO (R² >= {R2_THRESHOLD})")
    else:
        print(f"      ❌ RECHAZADO (R² < {R2_THRESHOLD})")
        passed = False
    
    return passed

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal del pipeline de validación"""
    try:
        # 1. Cargar datos
        X_test, y_test = load_test_data()
        
        # 2. Cargar modelo (con estrategia de fallback)
        model = load_trained_model()
        
        # 3. Evaluar modelo
        metrics = evaluate_model(model, X_test, y_test)
        
        # 4. Validar métricas
        passed = validate_metrics(metrics)
        
        # 5. Resultado final
        print("\n" + "="*70)
        if passed:
            print("✅ VALIDACIÓN EXITOSA - El modelo cumple todos los criterios")
            print("="*70)
            print("🚀 El modelo está listo para ser desplegado")
            return 0
        else:
            print("❌ VALIDACIÓN FALLIDA - El modelo NO cumple los criterios")
            print("="*70)
            print("🔄 Acciones recomendadas:")
            print("   • Revisar los datos de entrenamiento")
            print("   • Ajustar hiperparámetros del modelo")
            print("   • Considerar usar un modelo más complejo")
            print("   • Revisar si los umbrales son apropiados")
            return 1
            
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR EN LA VALIDACIÓN")
        print("="*70)
        traceback.print_exc()
        return 1

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
