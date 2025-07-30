# 📊 README Técnico - Análisis de Regresión Completo

## 🎯 Descripción del Proyecto

Este proyecto implementa un análisis completo de regresión para el dataset `dropout.csv`, enfocándose en predecir la **"Probabilidad Retirarse (1-5)"** usando diferentes modelos de regresión estadística.

## 🏗️ Arquitectura del Código

### 📁 Estructura Principal

```python
analisis_regresion_completo.py
├── cargar_datos()           # Carga del dataset
├── preprocesar_datos()      # Codificación de variables categóricas
├── regresion_multiple()     # Regresión múltiple (5 variables)
├── regresion_lineal_satisfaccion()  # Regresión lineal simple
├── regresion_curvilinea()   # Regresión exponencial
├── regresion_polinomial_grado4()    # Regresión polinomial
├── analisis_adicional()     # Correlaciones y estadísticas
├── crear_grafico_comparativo()      # Gráfico comparativo
└── main()                   # Función principal
```

## 🔧 Funciones Principales

### 1. **Carga y Preprocesamiento de Datos**

```python
def cargar_datos():
    """Carga el dataset dropout.csv."""
    try:
        df = pd.read_csv('dropout.csv')
        print(f"Dataset cargado exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'dropout.csv'")
        return None
```

```python
def preprocesar_datos(df):
    """Preprocesa los datos para el análisis de regresión."""
    # Variables categóricas a codificar
    variables_categoricas = ['Situación Laboral', 'Financiamiento Estudios', 
                           'Modalidad de Estudio', 'Considerado Dejar Carrera']
    
    # Codificar variables categóricas
    le = LabelEncoder()
    for col in variables_categoricas:
        if col in df_proc.columns:
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
            print(f"✓ Codificada: {col}")
    
    return df_proc
```

### 2. **Regresión Múltiple** (R² = 0.598)

```python
def regresion_multiple(df):
    """Implementa regresión múltiple para predecir Probabilidad Retirarse."""
    
    # Variables predictoras (5 variables)
    variables_predictoras = ['Índice Académico', 'Nivel Estrés Académico (1-5)', 
                           'Satisfacción Carrera (1-5)', 'Situación Laboral', 
                           'Considerado Dejar Carrera']
    
    # Preparar datos
    X = df[variables_disponibles].values
    y = df['Probabilidad Retirarse (1-5)'].values
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicciones y métricas
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Crear gráfico observado vs predicho
    plt.scatter(y, y_pred, alpha=0.6, s=60, color='blue', label='Datos')
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, 
             label='Predicción Perfecta')
```

**Resultados:**
- **R² = 0.598** (Mejor modelo)
- **MSE = 0.458**
- **Coeficientes:** Satisfacción (-0.478), Estrés (0.274), Índice (-0.156)

### 3. **Regresión Lineal Simple** (R² = 0.496)

```python
def regresion_lineal_satisfaccion(df):
    """Regresión lineal: Satisfacción Carrera vs Probabilidad Retirarse."""
    
    # Variables
    X_satisfaccion = df[['Satisfacción Carrera (1-5)']].values
    y_satisfaccion = df['Probabilidad Retirarse (1-5)'].values
    
    # Modelo y métricas
    model_satisfaccion = LinearRegression()
    model_satisfaccion.fit(X_satisfaccion, y_satisfaccion)
    
    # Estadísticas
    corr_satisfaccion, pval_satisfaccion = pearsonr(X_satisfaccion.flatten(), y_satisfaccion)
    r2_satisfaccion = r2_score(y_satisfaccion, y_pred_satisfaccion)
    
    # Gráfico con seaborn
    sns.regplot(x='Satisfacción Carrera (1-5)', y='Probabilidad Retirarse (1-5)', 
                data=df, ci=None, line_kws={'color':'red', 'linewidth': 2})
```

**Resultados:**
- **R² = 0.496** (2do mejor predictor)
- **r = -0.704** (Correlación negativa fuerte)
- **p = 6.42e-13** (Muy significativo)

### 4. **Regresión Curvilínea (Exponencial)** (R² = 0.443)

```python
def regresion_curvilinea(df):
    """Regresión exponencial: Estrés Académico vs Probabilidad Retirarse."""
    
    # Función exponencial: y = a * exp(b * x)
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    # Variables
    X_curve = df['Nivel Estrés Académico (1-5)'].values
    y_curve = df['Probabilidad Retirarse (1-5)'].values
    
    # Ajuste exponencial
    params, cov = curve_fit(exp_func, X_curve, y_curve, p0=(1, 0.1))
    y_pred_curve = exp_func(X_curve, *params)
    r2_curve = r2_score(y_curve, y_pred_curve)
    
    # Línea suave para visualización
    X_smooth = np.linspace(X_curve.min(), X_curve.max(), 100)
    y_smooth = exp_func(X_smooth, *params)
    plt.plot(X_smooth, y_smooth, color='red', linewidth=2, label='Ajuste Exponencial')
```

**Resultados:**
- **R² = 0.443** (3er mejor modelo)
- **Parámetros:** a=0.571, b=0.361
- **Relación:** Exponencial creciente

### 5. **Regresión Polinomial Grado 4** (R² = 0.222)

```python
def regresion_polinomial_grado4(df):
    """Regresión polinomial: Índice Académico vs Probabilidad Retirarse."""
    
    # Función polinomial: y = a*x^4 + b*x^3 + c*x^2 + d*x + e
    def poly4_func(x, a, b, c, d, e):
        return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e
    
    # Variables
    X_poly = df['Índice Académico'].values
    y_poly = df['Probabilidad Retirarse (1-5)'].values
    
    # Ajuste polinomial
    params, cov = curve_fit(poly4_func, X_poly, y_poly, p0=(0.5, -2, 3, -1, 4))
    y_pred_poly = poly4_func(X_poly, *params)
    r2_poly = r2_score(y_poly, y_pred_poly)
    
    # Comparación con otros modelos
    models = [("Polinomial Grado 4", r2_poly), ("Polinomial Grado 2", r2_poly2), 
             ("Lineal", r2_simple)]
    best_model = max(models, key=lambda x: x[1])
```

**Resultados:**
- **R² = 0.222** (Mejor entre polinomiales)
- **Función:** y = 4.200x⁴ + -34.850x³ + 103.725x² + -130.637x + 60.467
- **Comparación:** Grado 4 > Grado 2 > Lineal

## 📈 Conceptos Estadísticos Clave

### **Métricas de Evaluación**

```python
# R² (Coeficiente de determinación)
r2 = r2_score(y_true, y_pred)

# MSE (Error cuadrático medio)
mse = mean_squared_error(y_true, y_pred)

# Correlación de Pearson
corr, pval = pearsonr(x, y)

# Desviación estándar
std = np.std(x)
```

### **Tipos de Regresión Implementados**

| Modelo | R² | Variables | Comentario |
|--------|----|-----------|------------|
| **Múltiple** | 0.598 | 5 variables | Mejor modelo |
| **Lineal Satisfacción** | 0.496 | 1 variable | 2do mejor |
| **Curvilínea** | 0.443 | 1 variable | Exponencial |
| **Polinomial Grado 4** | 0.222 | 1 variable | Complejo |

### **Ranking de Predictores**

1. **Satisfacción Carrera** (r = -0.704, p = 6.42e-13)
2. **Estrés Académico** (r = 0.645, p = 1.82e-10)
3. **Índice Académico** (r = -0.365, p = 0.00102)

## 🎨 Visualización de Datos

### **Gráficos Generados**

```python
# Configuración de gráficos
plt.figure(figsize=(12, 8))
plt.title('Título del Gráfico\nR² = {r2:.3f} | r = {corr:.3f}')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.grid(True, alpha=0.3)

# Guardar gráfico
plt.savefig('nombre_grafico.png', dpi=300, bbox_inches='tight')
```

**Archivos PNG generados:**
- `regresion_multiple.png` (350KB)
- `regresion_lineal_satisfaccion.png` (253KB)
- `regresion_curvilinea.png` (218KB)
- `regresion_polinomial_grado4.png` (285KB)
- `comparacion_regresiones.png` (536KB)

## 🔍 Análisis de Correlaciones

```python
def analisis_adicional(df):
    """Análisis de correlaciones relevantes."""
    
    print("Resumen de correlaciones relevantes:")
    for col in ['Índice Académico', 'Nivel Estrés Académico (1-5)', 'Satisfacción Carrera (1-5)']:
        corr, pval = pearsonr(df[col], df['Probabilidad Retirarse (1-5)'])
        print(f"  {col} vs Probabilidad Retirarse: r={corr:.3f}, p={pval:.3g}")
    
    print("\nDesviaciones estándar de variables clave:")
    for col in ['Índice Académico', 'Nivel Estrés Académico (1-5)', 'Satisfacción Carrera (1-5)']:
        print(f"  {col}: {np.std(df[col]):.3f}")
```

## 🚀 Ejecución del Script

```bash
# Instalar dependencias
uv add pandas numpy matplotlib seaborn scipy scikit-learn

# Ejecutar análisis completo
uv run python analisis_regresion_completo.py
```

## 📊 Resultados Principales

### **Modelo Óptimo: Regresión Múltiple**
- **R² = 0.598** (59.8% de varianza explicada)
- **5 variables predictoras**
- **MSE = 0.458**

### **Mejor Predictor Individual: Satisfacción Carrera**
- **R² = 0.496** (49.6% de varianza explicada)
- **Correlación negativa fuerte** (r = -0.704)
- **Muy significativo** (p = 6.42e-13)

### **Interpretación de Resultados**
1. **Satisfacción con la carrera** es el factor más importante para predecir la probabilidad de retiro
2. **Estrés académico** tiene una relación exponencial con la probabilidad de retiro
3. **Índice académico** tiene una relación polinomial compleja
4. El **modelo múltiple** combina todos los factores de manera óptima

## 🛠️ Dependencias

```toml
# pyproject.toml
[project]
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0"
]
```

## 📝 Notas Técnicas

- **Codificación de variables categóricas** usando `LabelEncoder`
- **Manejo de errores** en ajustes de curvas con `try-except`
- **Visualización profesional** con estadísticas integradas
- **Comparación sistemática** de modelos
- **Documentación completa** de resultados

---

**Autor:** Asistente IA  
**Fecha:** 2024  
**Dataset:** dropout.csv (78 filas, 20 columnas) 