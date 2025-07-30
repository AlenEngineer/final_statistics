#!/usr/bin/env python3
"""
Análisis de Regresión Completo - Dataset Dropout
================================================

Este script implementa análisis de regresión para el dataset dropout.csv,
enfocándose en predecir la "Probabilidad Retirarse (1-5)" usando diferentes
modelos de regresión.

Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit

def cargar_datos():
    """
    Carga el dataset dropout.csv.
    
    Returns:
        pandas.DataFrame: Dataset cargado
    """
    try:
        df = pd.read_csv('dropout.csv')
        print(f"Dataset cargado exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'dropout.csv'")
        return None
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

def preprocesar_datos(df):
    """
    Preprocesa los datos para el análisis de regresión.
    
    Args:
        df (pandas.DataFrame): Dataset original
        
    Returns:
        pandas.DataFrame: Dataset preprocesado
    """
    print("\n=== PREPROCESAMIENTO DE DATOS ===")
    
    # Crear copia para no modificar el original
    df_proc = df.copy()
    
    # Identificar variables categóricas
    variables_categoricas = ['Situación Laboral', 'Financiamiento Estudios', 
                           'Modalidad de Estudio', 'Considerado Dejar Carrera']
    
    print(f"Variables categóricas encontradas: {variables_categoricas}")
    
    # Codificar variables categóricas
    le = LabelEncoder()
    for col in variables_categoricas:
        if col in df_proc.columns:
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
            print(f"✓ Codificada: {col}")
    
    return df_proc

def regresion_multiple(df):
    """
    Implementa regresión múltiple para predecir Probabilidad Retirarse.
    
    Args:
        df (pandas.DataFrame): Dataset preprocesado
    """
    print("\n=== REGRESIÓN MÚLTIPLE ===")
    
    # Variables predictoras
    variables_predictoras = ['Índice Académico', 'Nivel Estrés Académico (1-5)', 
                           'Satisfacción Carrera (1-5)', 'Situación Laboral', 
                           'Considerado Dejar Carrera']
    
    # Filtrar variables disponibles
    variables_disponibles = [var for var in variables_predictoras if var in df.columns]
    print(f"Variables predictoras utilizadas: {variables_disponibles}")
    
    # Preparar datos
    X = df[variables_disponibles].values
    y = df['Probabilidad Retirarse (1-5)'].values
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicciones
    y_pred = model.predict(X)
    
    # Métricas
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Crear gráfico de valores observados vs predichos
    plt.figure(figsize=(12, 8))
    plt.scatter(y, y_pred, alpha=0.6, s=60, color='blue', label='Datos')
    
    # Línea de predicción perfecta
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Predicción Perfecta')
    
    plt.title('Regresión Múltiple: Probabilidad Retirarse Observada vs Predicha\n'
              f'Modelo con {len(variables_disponibles)} variables predictoras | R² = {r2:.3f} | MSE = {mse:.3f}')
    plt.xlabel('Probabilidad Retirarse (1-5) - Observada')
    plt.ylabel('Probabilidad Retirarse (1-5) - Predicha')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Agregar anotación con coeficientes
    coef_text = "Coeficientes:\n"
    for i, var in enumerate(variables_disponibles):
        coef_text += f"  {var}: {model.coef_[i]:.3f}\n"
    coef_text += f"  Intercepto: {model.intercept_:.3f}"
    
    plt.text(0.02, 0.98, coef_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Guardar gráfico
    plt.savefig('regresion_multiple.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado como 'regresion_multiple.png'")
    
    try:
        plt.show()
    except:
        print("⚠️  No se puede mostrar gráfico interactivo en este entorno")
    
    # Imprimir resultados
    print(f"R²: {r2:.3f}")
    print(f"MSE: {mse:.3f}")
    print("Coeficientes:")
    for name, coef in zip(variables_disponibles, model.coef_):
        print(f"  {name}: {coef:.3f}")
    print(f"Intercepto: {model.intercept_:.3f}")

def regresion_lineal_satisfaccion(df):
    """
    Implementa regresión lineal simple para Satisfacción en la Carrera vs Índice Académico.
    
    Args:
        df (pandas.DataFrame): Dataset preprocesado
    """
    print("\n=== REGRESIÓN LINEAL SIMPLE: SATISFACCIÓN EN LA CARRERA ===")
    
    # Seleccionar variables: Satisfacción en la Carrera vs Probabilidad Retirarse
    X_satisfaccion = df[['Satisfacción Carrera (1-5)']].values
    y_satisfaccion = df['Probabilidad Retirarse (1-5)'].values
    
    # Entrenar modelo de regresión lineal
    model_satisfaccion = LinearRegression()
    model_satisfaccion.fit(X_satisfaccion, y_satisfaccion)
    
    # Predicciones
    y_pred_satisfaccion = model_satisfaccion.predict(X_satisfaccion)
    
    # Métricas
    r2_satisfaccion = r2_score(y_satisfaccion, y_pred_satisfaccion)
    mse_satisfaccion = mean_squared_error(y_satisfaccion, y_pred_satisfaccion)
    
    # Calcular correlación y estadísticas
    corr_satisfaccion, pval_satisfaccion = pearsonr(X_satisfaccion.flatten(), y_satisfaccion)
    std_satisfaccion = np.std(X_satisfaccion)
    
    # Crear gráfico sin banda de confianza
    plt.figure(figsize=(12, 8))
    sns.regplot(x='Satisfacción Carrera (1-5)', y='Probabilidad Retirarse (1-5)', data=df, 
                ci=None, line_kws={'color':'red', 'linewidth': 2})
    plt.title('Regresión Lineal Simple: Satisfacción en la Carrera vs Probabilidad Retirarse\n'
              f'R² = {r2_satisfaccion:.3f} | r = {corr_satisfaccion:.3f} (p = {pval_satisfaccion:.3g}) | σ = {std_satisfaccion:.3f}')
    plt.xlabel('Satisfacción Carrera (1-5)')
    plt.ylabel('Probabilidad Retirarse (1-5)')
    plt.grid(True, alpha=0.3)
    
    # Agregar anotación con estadísticas
    stats_text = f"R² = {r2_satisfaccion:.3f}\n"
    stats_text += f"r = {corr_satisfaccion:.3f}\n"
    stats_text += f"p = {pval_satisfaccion:.3g}\n"
    stats_text += f"σ = {std_satisfaccion:.3f}\n"
    stats_text += f"Coef = {model_satisfaccion.coef_[0]:.3f}\n"
    stats_text += f"Intercept = {model_satisfaccion.intercept_:.3f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12)
    
    # Guardar gráfico
    plt.savefig('regresion_lineal_satisfaccion.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado como 'regresion_lineal_satisfaccion.png'")
    
    try:
        plt.show()
    except:
        print("⚠️  No se puede mostrar gráfico interactivo en este entorno")
    
    # Imprimir resultados
    print(f"Correlación Pearson: {corr_satisfaccion:.3f} (p={pval_satisfaccion:.3g})")
    print(f"Desviación estándar de la Satisfacción: {std_satisfaccion:.3f}")
    print(f"R² del modelo: {r2_satisfaccion:.3f}")
    print(f"Coeficiente: {model_satisfaccion.coef_[0]:.3f}")
    print(f"Intercepto: {model_satisfaccion.intercept_:.3f}")
    print(f"MSE: {mse_satisfaccion:.3f}")

def regresion_curvilinea(df):
    """
    Implementa regresión curvilínea (exponencial) para Estrés Académico vs Probabilidad Retirarse.
    
    Args:
        df (pandas.DataFrame): Dataset preprocesado
    """
    print("\n=== REGRESIÓN CURVILÍNEA (EXPONENCIAL) ===")
    
    # Función exponencial: y = a * exp(b * x)
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    # Seleccionar variables: Estrés Académico vs Probabilidad Retirarse
    X_curve = df['Nivel Estrés Académico (1-5)'].values
    y_curve = df['Probabilidad Retirarse (1-5)'].values
    
    # Ajuste exponencial
    try:
        params, cov = curve_fit(exp_func, X_curve, y_curve, p0=(1, 0.1))
        y_pred_curve = exp_func(X_curve, *params)
        r2_curve = r2_score(y_curve, y_pred_curve)
        
        # Crear gráfico con estadísticas
        plt.figure(figsize=(12, 8))
        plt.scatter(X_curve, y_curve, alpha=0.6, label='Datos', s=60, color='blue')
        
        # Crear línea suave para la regresión curvilínea
        X_smooth = np.linspace(X_curve.min(), X_curve.max(), 100)
        y_smooth = exp_func(X_smooth, *params)
        plt.plot(X_smooth, y_smooth, color='red', linewidth=2, label='Ajuste Exponencial')
        
        plt.title('Regresión Curvilínea: Estrés Académico vs Probabilidad Retirarse\n'
                  f'R² = {r2_curve:.3f} | Parámetros: a={params[0]:.3f}, b={params[1]:.3f}')
        plt.xlabel('Nivel Estrés Académico (1-5)')
        plt.ylabel('Probabilidad Retirarse (1-5)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calcular correlación y estadísticas
        corr_curve, pval_curve = pearsonr(X_curve, y_curve)
        std_curve = np.std(X_curve)
        
        # Agregar anotación con estadísticas
        stats_text = f"R² = {r2_curve:.3f}\n"
        stats_text += f"r = {corr_curve:.3f}\n"
        stats_text += f"p = {pval_curve:.3g}\n"
        stats_text += f"σ = {std_curve:.3f}\n"
        stats_text += f"a = {params[0]:.3f}\n"
        stats_text += f"b = {params[1]:.3f}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=12)
        
        # Guardar gráfico
        plt.savefig('regresion_curvilinea.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico guardado como 'regresion_curvilinea.png'")
        
        try:
            plt.show()
        except:
            print("⚠️  No se puede mostrar gráfico interactivo en este entorno")
        
        # Imprimir resultados
        print(f"Parámetros: a={params[0]:.3f}, b={params[1]:.3f}")
        print(f"R²: {r2_curve:.3f}")
        
    except Exception as e:
        print(f"Error en el ajuste exponencial: {e}")

def regresion_polinomial_grado4(df):
    """
    Implementa regresión polinomial de cuarto grado para Índice Académico vs Probabilidad Retirarse.
    
    Args:
        df (pandas.DataFrame): Dataset preprocesado
    """
    print("\n=== REGRESIÓN POLINOMIAL (GRADO 4) ===")
    
    # Función polinomial de grado 4: y = a*x^4 + b*x^3 + c*x^2 + d*x + e
    def poly4_func(x, a, b, c, d, e):
        return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e
    
    # Seleccionar variables: Índice Académico vs Probabilidad Retirarse
    X_poly = df['Índice Académico'].values
    y_poly = df['Probabilidad Retirarse (1-5)'].values
    
    # Ajuste polinomial de grado 4
    try:
        params, cov = curve_fit(poly4_func, X_poly, y_poly, p0=(0.5, -2, 3, -1, 4))
        y_pred_poly = poly4_func(X_poly, *params)
        r2_poly = r2_score(y_poly, y_pred_poly)
        
        # Crear gráfico con estadísticas
        plt.figure(figsize=(12, 8))
        plt.scatter(X_poly, y_poly, alpha=0.6, label='Datos', s=60, color='blue')
        
        # Crear línea suave para la regresión polinomial
        X_smooth = np.linspace(X_poly.min(), X_poly.max(), 200)
        y_smooth = poly4_func(X_smooth, *params)
        plt.plot(X_smooth, y_smooth, color='red', linewidth=2, label='Ajuste Polinomial Grado 4')
        
        plt.title('Regresión Polinomial Grado 4: Índice Académico vs Probabilidad Retirarse\n'
                  f'R² = {r2_poly:.3f} | Función: y = {params[0]:.3f}x⁴ + {params[1]:.3f}x³ + {params[2]:.3f}x² + {params[3]:.3f}x + {params[4]:.3f}')
        plt.xlabel('Índice Académico')
        plt.ylabel('Probabilidad Retirarse (1-5)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calcular correlación y estadísticas
        corr_poly, pval_poly = pearsonr(X_poly, y_poly)
        std_poly = np.std(X_poly)
        
        # Agregar anotación con estadísticas
        stats_text = f"R² = {r2_poly:.3f}\n"
        stats_text += f"r = {corr_poly:.3f}\n"
        stats_text += f"p = {pval_poly:.3g}\n"
        stats_text += f"σ = {std_poly:.3f}\n"
        stats_text += f"a = {params[0]:.3f}\n"
        stats_text += f"b = {params[1]:.3f}\n"
        stats_text += f"c = {params[2]:.3f}\n"
        stats_text += f"d = {params[3]:.3f}\n"
        stats_text += f"e = {params[4]:.3f}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=10)
        
        # Guardar gráfico
        plt.savefig('regresion_polinomial_grado4.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico guardado como 'regresion_polinomial_grado4.png'")
        
        try:
            plt.show()
        except:
            print("⚠️  No se puede mostrar gráfico interactivo en este entorno")
        
        # Imprimir resultados
        print(f"Función: y = {params[0]:.3f}x⁴ + {params[1]:.3f}x³ + {params[2]:.3f}x² + {params[3]:.3f}x + {params[4]:.3f}")
        print(f"R²: {r2_poly:.3f}")
        print(f"Coeficientes: a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}, d={params[3]:.3f}, e={params[4]:.3f}")
        
        # Comparar con otros modelos
        # Regresión lineal simple
        X_simple = df[['Índice Académico']].values
        y_simple = df['Probabilidad Retirarse (1-5)'].values
        model_simple = LinearRegression()
        model_simple.fit(X_simple, y_simple)
        y_pred_simple = model_simple.predict(X_simple)
        r2_simple = r2_score(y_simple, y_pred_simple)
        
        # Polinomial grado 2
        def poly2_func(x, a, b, c):
            return a * np.power(x, 2) + b * x + c
        try:
            params_poly2, _ = curve_fit(poly2_func, X_poly, y_poly, p0=(1, -1, 3))
            y_pred_poly2 = poly2_func(X_poly, *params_poly2)
            r2_poly2 = r2_score(y_poly, y_pred_poly2)
        except:
            r2_poly2 = 0.0
        
        print(f"\nComparación de Modelos:")
        print(f"  Polinomial Grado 4 R²: {r2_poly:.3f}")
        print(f"  Polinomial Grado 2 R²: {r2_poly2:.3f}")
        print(f"  Lineal Simple R²: {r2_simple:.3f}")
        
        # Determinar el mejor modelo
        models = [("Polinomial Grado 4", r2_poly), ("Polinomial Grado 2", r2_poly2), 
                 ("Lineal", r2_simple)]
        best_model = max(models, key=lambda x: x[1])
        
        print(f"\n🏆 Mejor modelo: {best_model[0]} (R² = {best_model[1]:.3f})")
        
        # Análisis de la forma de la curva
        print(f"\nAnálisis de la forma de la curva:")
        print(f"  Coeficiente de mayor grado (a): {params[0]:.3f}")
        if params[0] > 0:
            print(f"  📈 Término x⁴ positivo: Curva tiende hacia arriba en los extremos")
        else:
            print(f"  📉 Término x⁴ negativo: Curva tiende hacia abajo en los extremos")
        print(f"  🔍 Análisis de puntos críticos y forma de la curva")
        
    except Exception as e:
        print(f"Error en el ajuste polinomial grado 4: {e}")
        print("Intentando con polinomio de grado 5...")
        
        # Fallback a polinomio de grado 5
        def poly5_func(x, a, b, c, d, e, f):
            return a * np.power(x, 5) + b * np.power(x, 4) + c * np.power(x, 3) + d * np.power(x, 2) + e * x + f
        
        try:
            params_poly5, cov_poly5 = curve_fit(poly5_func, X_poly, y_poly, p0=(0.1, -1, 2, -1, 0, 3))
            y_pred_poly5 = poly5_func(X_poly, *params_poly5)
            r2_poly5 = r2_score(y_poly, y_pred_poly5)
            print(f"Polinomio de grado 5: R² = {r2_poly5:.3f}")
        except Exception as e2:
            print(f"Error también con polinomio de grado 5: {e2}")
            print("Recomendación: Usar función logística o mantener polinomio de grado 2")

def analisis_adicional(df):
    """
    Realiza análisis adicional de correlaciones y estadísticas.
    
    Args:
        df (pandas.DataFrame): Dataset preprocesado
    """
    print("\n=== ANÁLISIS ADICIONAL ===")
    
    # Correlaciones relevantes
    print("Resumen de correlaciones relevantes:")
    for col in ['Índice Académico', 'Nivel Estrés Académico (1-5)', 'Satisfacción Carrera (1-5)']:
        corr, pval = pearsonr(df[col], df['Probabilidad Retirarse (1-5)'])
        print(f"  {col} vs Probabilidad Retirarse: r={corr:.3f}, p={pval:.3g}")
    
    # Desviaciones estándar
    print("\nDesviaciones estándar de variables clave:")
    for col in ['Índice Académico', 'Nivel Estrés Académico (1-5)', 'Satisfacción Carrera (1-5)']:
        print(f"  {col}: {np.std(df[col]):.3f}")

def crear_grafico_comparativo(df):
    """
    Crea un gráfico comparativo de los diferentes modelos de regresión.
    
    Args:
        df (pandas.DataFrame): Dataset preprocesado
    """
    print("\n=== GRÁFICO COMPARATIVO ===")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparación de Modelos de Regresión', fontsize=18)
    
    # 1. Regresión Múltiple
    ax1 = axes[0, 0]
    variables_predictoras = ['Índice Académico', 'Nivel Estrés Académico (1-5)', 
                           'Satisfacción Carrera (1-5)', 'Situación Laboral', 
                           'Considerado Dejar Carrera']
    variables_disponibles = [var for var in variables_predictoras if var in df.columns]
    X_multi = df[variables_disponibles].values
    y = df['Probabilidad Retirarse (1-5)'].values
    model_multi = LinearRegression()
    model_multi.fit(X_multi, y)
    y_pred_multi = model_multi.predict(X_multi)
    r2_multi = r2_score(y, y_pred_multi)
    
    ax1.scatter(y, y_pred_multi, alpha=0.6, s=40)
    min_val = min(y.min(), y_pred_multi.min())
    max_val = max(y.max(), y_pred_multi.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    ax1.set_title(f'Regresión Múltiple\nR² = {r2_multi:.3f}')
    ax1.set_xlabel('Observado')
    ax1.set_ylabel('Predicho')
    ax1.grid(True, alpha=0.3)
    
    # 2. Regresión Curvilínea
    ax2 = axes[0, 1]
    X_curve = df['Nivel Estrés Académico (1-5)'].values
    y_curve = df['Probabilidad Retirarse (1-5)'].values
    
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    try:
        params, _ = curve_fit(exp_func, X_curve, y_curve, p0=(1, 0.1))
        y_pred_curve = exp_func(X_curve, *params)
        r2_curve = r2_score(y_curve, y_pred_curve)
        
        ax2.scatter(X_curve, y_curve, alpha=0.6, s=40)
        X_smooth = np.linspace(X_curve.min(), X_curve.max(), 100)
        y_smooth = exp_func(X_smooth, *params)
        ax2.plot(X_smooth, y_smooth, 'r-', linewidth=2)
        ax2.set_title(f'Regresión Curvilínea\nR² = {r2_curve:.3f}')
        ax2.set_xlabel('Estrés Académico')
        ax2.set_ylabel('Probabilidad Retirarse')
        ax2.grid(True, alpha=0.3)
    except:
        ax2.text(0.5, 0.5, 'Error en ajuste', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Regresión Curvilínea\nError en ajuste')
    
    # 3. Regresión Polinomial Grado 4
    ax3 = axes[1, 0]
    X_poly = df['Índice Académico'].values
    y_poly = df['Probabilidad Retirarse (1-5)'].values
    
    def poly4_func(x, a, b, c, d, e):
        return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e
    
    try:
        params_poly, _ = curve_fit(poly4_func, X_poly, y_poly, p0=(0.5, -2, 3, -1, 4))
        y_pred_poly = poly4_func(X_poly, *params_poly)
        r2_poly = r2_score(y_poly, y_pred_poly)
        
        ax3.scatter(X_poly, y_poly, alpha=0.6, s=40)
        X_smooth = np.linspace(X_poly.min(), X_poly.max(), 200)
        y_smooth = poly4_func(X_smooth, *params_poly)
        ax3.plot(X_smooth, y_smooth, 'r-', linewidth=2)
        ax3.set_title(f'Polinomial Grado 4\nR² = {r2_poly:.3f}')
        ax3.set_xlabel('Índice Académico')
        ax3.set_ylabel('Probabilidad Retirarse')
        ax3.grid(True, alpha=0.3)
    except:
        ax3.text(0.5, 0.5, 'Error en ajuste', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Polinomial Grado 4\nError en ajuste')
    
    # 4. Comparación de R²
    ax4 = axes[1, 1]
    modelos = ['Múltiple', 'Curvilínea', 'Polinomial Grado 4']
    
    # Calcular R² reales
    # 1. Regresión Múltiple
    X_simple = df[['Índice Académico']].values
    y = df['Probabilidad Retirarse (1-5)'].values
    model_simple = LinearRegression()
    model_simple.fit(X_simple, y)
    y_pred_simple = model_simple.predict(X_simple)
    r2_simple_real = r2_score(y, y_pred_simple)
    
    # 2. Regresión Múltiple
    variables_predictoras = ['Índice Académico', 'Nivel Estrés Académico (1-5)', 
                           'Satisfacción Carrera (1-5)', 'Situación Laboral', 
                           'Considerado Dejar Carrera']
    variables_disponibles = [var for var in variables_predictoras if var in df.columns]
    X_multi = df[variables_disponibles].values
    model_multi = LinearRegression()
    model_multi.fit(X_multi, y)
    y_pred_multi = model_multi.predict(X_multi)
    r2_multi_real = r2_score(y, y_pred_multi)
    
    # 3. Regresión Curvilínea
    X_curve = df['Nivel Estrés Académico (1-5)'].values
    y_curve = df['Probabilidad Retirarse (1-5)'].values
    
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    try:
        params, cov = curve_fit(exp_func, X_curve, y_curve, p0=(1, 0.1))
        y_pred_curve = exp_func(X_curve, *params)
        r2_curve_real = r2_score(y_curve, y_pred_curve)
    except:
        r2_curve_real = 0.0
    
    # 4. Polinomial Grado 4
    X_poly = df['Índice Académico'].values
    y_poly = df['Probabilidad Retirarse (1-5)'].values
    
    def poly4_func(x, a, b, c, d, e):
        return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e
    
    try:
        params_poly4, _ = curve_fit(poly4_func, X_poly, y_poly, p0=(0.5, -2, 3, -1, 4))
        y_pred_poly4 = poly4_func(X_poly, *params_poly4)
        r2_poly4_real = r2_score(y_poly, y_pred_poly4)
    except:
        r2_poly4_real = 0.0
    
    r2_values = [r2_multi_real, r2_curve_real, r2_poly4_real]
    
    bars = ax4.bar(modelos, r2_values, color=['red', 'blue', 'green'])
    ax4.set_title('Comparación de R² de los Tres Modelos')
    ax4.set_ylabel('R²')
    ax4.set_ylim(0, 1)
    
    # Agregar valores en las barras
    for bar, value in zip(bars, r2_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Agregar línea de referencia en 0.5
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='R² = 0.5')
    ax4.legend()
    
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig('comparacion_regresiones.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico comparativo guardado como 'comparacion_regresiones.png'")
    
    try:
        plt.show()
    except:
        print("⚠️  No se puede mostrar gráfico interactivo en este entorno")
    
    # Información adicional
    print(f"\n📊 Variables utilizadas en Regresión Múltiple ({len(variables_disponibles)} variables):")
    for i, var in enumerate(variables_disponibles, 1):
        print(f"  {i}. {var}")
    print(f"  Variable objetivo: Probabilidad Retirarse (1-5)")
    print(f"  R² obtenido: {r2_multi_real:.3f}")

def main():
    """
    Función principal que ejecuta todo el análisis de regresión.
    """
    print("=" * 70)
    print("ANÁLISIS DE REGRESIÓN COMPLETO - DATASET DROPOUT")
    print("=" * 70)
    
    # 1. Cargar datos
    df = cargar_datos()
    if df is None:
        return
    
    # 2. Preprocesar datos
    df_proc = preprocesar_datos(df)
    
    # 3. Regresión Múltiple
    regresion_multiple(df_proc)
    
    # 4. Regresión Lineal Simple: Satisfacción en la Carrera
    regresion_lineal_satisfaccion(df_proc)
    
    # 5. Regresión Curvilínea (Exponencial)
    regresion_curvilinea(df_proc)
    
    # 5. Regresión Polinomial (Grado 4)
    regresion_polinomial_grado4(df_proc)
    
    # 6. Análisis Adicional
    analisis_adicional(df_proc)
    
    # 7. Gráfico Comparativo
    crear_grafico_comparativo(df_proc)
    
    print("\n" + "=" * 70)
    print("ANÁLISIS DE REGRESIÓN COMPLETADO")
    print("=" * 70)
    
    print("\nArchivos generados:")
    print("- regresion_multiple.png: Regresión múltiple")
    print("- regresion_lineal_satisfaccion.png: Regresión lineal simple (Satisfacción)")
    print("- regresion_curvilinea.png: Regresión curvilínea")
    print("- regresion_polinomial_grado4.png: Regresión polinomial (Grado 4)")
    print("- comparacion_regresiones.png: Gráfico comparativo")
    print("\nTipos de regresión implementados:")
    print("1. Regresión Múltiple")
    print("2. Regresión Lineal Simple (Satisfacción)")
    print("3. Regresión Curvilínea (Exponencial)")
    print("4. Regresión Polinomial (Grado 4)")

if __name__ == "__main__":
    main() 