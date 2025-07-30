# Análisis de Regresión - Dataset Dropout

## 📊 Descripción del Proyecto

Este proyecto implementa un análisis de regresión completo para el dataset `dropout.csv`, enfocándose en predecir la "Probabilidad Retirarse (1-5)" usando diferentes modelos de regresión. El análisis incluye regresión múltiple, lineal simple, curvilínea y polinomial.

## 🎯 Objetivos

- Analizar factores que influyen en la probabilidad de retirarse de la carrera
- Comparar diferentes modelos de regresión
- Identificar las variables más importantes para predecir el abandono académico
- Generar visualizaciones profesionales de los resultados

## 📁 Estructura del Proyecto

```
final/
├── dropout.csv                    # Dataset principal
├── analisis_regresion_completo.py # Script principal de análisis
├── pyproject.toml                # Configuración de dependencias
├── README.md                     # Este archivo
└── *.png                        # Gráficos generados
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- `uv` (gestor de paquetes)

### Instalación
```bash
# Clonar el repositorio (si aplica)
git clone <repository-url>
cd final

# Instalar dependencias con uv
uv add pandas numpy matplotlib seaborn scipy scikit-learn
```

## 📈 Tipos de Regresión Implementados

### 1. Regresión Múltiple
- **Variables predictoras:** 5 variables (Índice Académico, Estrés Académico, Satisfacción Carrera, Situación Laboral, Considerado Dejar Carrera)
- **Variable objetivo:** Probabilidad Retirarse (1-5)
- **R²:** 0.598 (59.8% de varianza explicada)
- **Archivo:** `regresion_multiple.png`

### 2. Regresión Lineal Simple (Satisfacción)
- **Variables:** Satisfacción Carrera (1-5) vs Probabilidad Retirarse (1-5)
- **R²:** 0.496 (49.6% de varianza explicada)
- **Correlación:** r = -0.704 (p < 0.001)
- **Interpretación:** Mayor satisfacción → Menor probabilidad de retirarse
- **Archivo:** `regresion_lineal_satisfaccion.png`

### 3. Regresión Curvilínea (Exponencial)
- **Variables:** Estrés Académico (1-5) vs Probabilidad Retirarse (1-5)
- **Modelo:** y = a * exp(b * x)
- **R²:** 0.443 (44.3% de varianza explicada)
- **Parámetros:** a=0.571, b=0.361
- **Archivo:** `regresion_curvilinea.png`

### 4. Regresión Polinomial (Grado 4)
- **Variables:** Índice Académico vs Probabilidad Retirarse (1-5)
- **Modelo:** y = ax⁴ + bx³ + cx² + dx + e
- **R²:** 0.222 (22.2% de varianza explicada)
- **Coeficientes:** a=4.200, b=-34.850, c=103.725, d=-130.637, e=60.467
- **Archivo:** `regresion_polinomial_grado4.png`

## 📊 Resultados Principales

### Ranking de Predictores de Probabilidad Retirarse

| Posición | Variable | R² | Correlación | Interpretación |
|----------|----------|----|-------------|----------------|
| 🥇 **1er** | **Satisfacción Carrera** | **0.496** | r = -0.704 | **Mejor predictor** |
| 🥈 **2do** | **Estrés Académico** | 0.416 | r = 0.645 | Relación positiva fuerte |
| 🥉 **3er** | **Índice Académico** | 0.133 | r = -0.365 | Relación negativa moderada |

### Modelo General Más Efectivo
- **Regresión Múltiple:** R² = 0.598 (59.8%)
- **Variables utilizadas:** 5 predictoras
- **Mejor modelo para predicción general**

## 🎨 Gráficos Generados

### Archivos PNG Producidos:
1. **`regresion_multiple.png`** (350KB) - Regresión múltiple con 5 variables
2. **`regresion_lineal_satisfaccion.png`** (253KB) - Regresión lineal satisfacción vs probabilidad retirarse
3. **`regresion_curvilinea.png`** (218KB) - Regresión exponencial estrés vs probabilidad retirarse
4. **`regresion_polinomial_grado4.png`** (285KB) - Regresión polinomial índice vs probabilidad retirarse
5. **`comparacion_regresiones.png`** (536KB) - Gráfico comparativo de todos los modelos

### Características de los Gráficos:
- ✅ **Banda de confianza** (donde aplica)
- ✅ **Estadísticas completas** en anotaciones
- ✅ **Títulos descriptivos** con métricas
- ✅ **Etiquetas claras** de ejes
- ✅ **Cuadrícula** para mejor visualización

## 🔧 Uso del Script

### Ejecutar Análisis Completo
```bash
uv run python analisis_regresion_completo.py
```

### Funciones Principales
- `cargar_datos()`: Carga el dataset dropout.csv
- `preprocesar_datos()`: Codifica variables categóricas
- `regresion_multiple()`: Regresión múltiple con 5 variables
- `regresion_lineal_satisfaccion()`: Regresión lineal satisfacción vs probabilidad retirarse
- `regresion_curvilinea()`: Regresión exponencial
- `regresion_polinomial_grado4()`: Regresión polinomial grado 4
- `analisis_adicional()`: Correlaciones y estadísticas adicionales
- `crear_grafico_comparativo()`: Gráfico comparativo de modelos

## 📋 Variables del Dataset

### Variables Numéricas:
- **Índice Académico:** Rendimiento académico del estudiante
- **Nivel Estrés Académico (1-5):** Nivel de estrés percibido
- **Satisfacción Carrera (1-5):** Satisfacción con la carrera elegida
- **Probabilidad Retirarse (1-5):** Variable objetivo - probabilidad de abandonar

### Variables Categóricas (Codificadas):
- **Situación Laboral:** Estado laboral del estudiante
- **Financiamiento Estudios:** Tipo de financiamiento
- **Modalidad de Estudio:** Modalidad de estudio
- **Considerado Dejar Carrera:** Frecuencia de considerar abandonar

## 📈 Interpretación de Resultados

### Hallazgos Principales:
1. **La satisfacción con la carrera es el factor más importante** para predecir la probabilidad de retirarse (R² = 0.496)
2. **El estrés académico tiene una relación positiva fuerte** con la probabilidad de retirarse (R² = 0.416)
3. **El índice académico tiene una relación negativa moderada** con la probabilidad de retirarse (R² = 0.133)
4. **El modelo múltiple es el más efectivo** para predicción general (R² = 0.598)

### Recomendaciones:
- **Mejorar la satisfacción estudiantil** es la estrategia más efectiva para reducir el abandono
- **Gestionar el estrés académico** es crucial para retener estudiantes
- **El rendimiento académico** tiene un impacto moderado en la decisión de retirarse

## 🛠️ Dependencias

```toml
# pyproject.toml
[project]
dependencies = [
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "scipy>=1.9.0",
    "scikit-learn>=1.1.0"
]
```

## 📝 Autor

- **Proyecto:** Análisis de Regresión - Dataset Dropout
- **Fecha:** 2024
- **Herramientas:** Python, pandas, matplotlib, seaborn, scikit-learn

## 📄 Licencia

Este proyecto es para fines educativos y de investigación.

---

**Nota:** Todos los gráficos se generan automáticamente al ejecutar el script principal. Los archivos PNG contienen visualizaciones profesionales con estadísticas completas y anotaciones detalladas.
