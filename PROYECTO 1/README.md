# Proyecto: Naive Bayes con Estimación KDE para Mantenimiento Predictivo

**Curso:** IA / Mantenimiento Predictivo  
**Dataset:** `ai4i2020.csv` (AI4I 2020 Predictive Maintenance Dataset)  
**Objetivo:** Predecir la variable **`Machine failure`** (0/1) comparando dos enfoques:
1) **Gaussian Naive Bayes (GNB)** – suposición paramétrica (distribución normal).  
2) **KDE Naive Bayes (KDE-NB)** – estimación **no paramétrica** de densidades (KDE) + ajuste de umbral.

---

## 1) Estructura del proyecto

```
PROYECTO 1/
├─ ai4i2020.csv
├─ proyecto_1.ipynb              # Notebook principal
├─ requirements.txt              # Dependencias
├─ README.md                     # Este archivo
├─ .vscode/settings.json         # (Opcional) VS Code apuntando a .venv
├─ figs/                         # Gráficos exportados
├─ cm_gnb_05.csv                 # Matriz de confusión GNB (0.5)
├─ cm_kde_05.csv                 # Matriz de confusión KDE-NB (0.5)
├─ cm_kde_opt.csv                # Matriz de confusión KDE-NB (umbral óptimo)
├─ comparativa_nb.csv            # Tabla comparativa de métricas
├─ cv_auc.csv                    # (Nuevo) AUC por fold (GNB y KDE-NB)
└─ comparativa_tiempos.csv       # (Nuevo) Tiempos de cómputo
```

> Las carpetas/archivos marcados como “(Nuevo)” se generan al ejecutar las celdas añadidas abajo.

---

## 2) Configuración rápida (Windows + VS Code)

**Python 3.10–3.12**. Abre `PROYECTO 1` como carpeta de trabajo.

```powershell
# Crear venv (si no existe) y activarlo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Si PowerShell bloquea scripts: 
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) Columnas usadas

**Target (y):** `Machine failure` (0 = no falla, 1 = falla).  

**Features numéricas:** `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.  
**Categórica:** `Type` → One-Hot (`Type_L`, `Type_M`, `Type_H`).  
**No usar (ID/fuga):** `UDI`, `Product ID`, y `TWF/HDF/PWF/OSF/RNF`.

---

## 4) Flujo del notebook `proyecto_1.ipynb`

1. **Carga & limpieza mínima** del CSV  
2. **Construcción de `X` y `y`** (numéricas + One-Hot)  
3. **Split estratificado 80/20**  
4. **Baseline – GNB** (accuracy, reporte, CM → `cm_gnb_05.csv`)  
5. **KDE-NB** (probabilidades, evaluación 0.5 → `cm_kde_05.csv`)  
6. **Ajuste de umbral por F1(positiva)** y reevaluación (→ `cm_kde_opt.csv`)  
7. **Comparativa** (accuracy, precision/recall/F1 → `comparativa_nb.csv`)  
8. **Gráficos**: PR/ROC, matrices de confusión, distribuciones y correlación (guardados en `figs/`)

---

## 5) Cómo ejecutar

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook   # o "jupyter lab"
```

Ejecuta el notebook **de arriba a abajo**. Al terminar tendrás CSVs y figuras en `figs/`.

---

## 6) Resultados esperados (guía)

- **GNB (0.5):** buena accuracy, **F1 de clase 1** moderada por desbalance.  
- **KDE-NB (0.5):** suele mejorar **recall/F1** de clase 1 vs GNB.  
- **KDE-NB (umbral óptimo):** mejora equilibrio precision–recall para fallas, con poca pérdida de accuracy global.

---

## 7) Buenas prácticas y siguientes pasos

- Reportar accuracy + precision/recall/F1 (macro/weighted y por clase).  
- En desbalance, priorizar **Precision-Recall** y **F1 de la clase 1**.  
- Validar con **K-Fold estratificado** o **TimeSeriesSplit** si hay temporalidad.  
- Explorar kernels/bandwidths en KDE; calibración (`CalibratedClassifierCV`).  
- Probar modelos adicionales (LogReg, RF, XGBoost/LightGBM).  
- Ingeniería de variables y *scaling* si aplica.

---

## 8) Solución de problemas

- **Venv / scripts en PowerShell:**  
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y `.\.venv\Scripts\Activate.ps1`.
- **“Import could not be resolved” en VS Code:**  
  1) Kernel = `.venv\Scripts\python.exe`, 2) `pip install -r requirements.txt`, 3) reinicia VS Code.

---

## 9) **Validación cruzada 5-Fold y AUC (curva ROC)**

**Objetivo (entregable):** Reportar **AUC promedio** tras **≥5 folds** para GNB y KDE-NB.  
Añade una celda al notebook con:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np, pandas as pd, os

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def proba_pos(model, X):
    # model con predict_proba
    return model.predict_proba(X)[:,1]

auc_rows = []

for name, make_model in [
    ("GNB", lambda: GaussianNB()),
    ("KDE-NB", lambda: KDENaiveBayes())   # tu clase/implementación KDE-NB
]:
    aucs = []
    for tr, va in skf.split(X, y):
        m = make_model().fit(X.iloc[tr].values, y.iloc[tr].values)
        p = proba_pos(m, X.iloc[va].values)
        aucs.append(roc_auc_score(y.iloc[va].values, p))
    auc_rows.append({"modelo": name, "AUC_mean": np.mean(aucs), "AUC_std": np.std(aucs), "folds": len(aucs)})

cv_auc = pd.DataFrame(auc_rows)
cv_auc.to_csv("cv_auc.csv", index=False)
cv_auc
```

> **Figuras para la presentación**: genera también **curvas ROC** por modelo en `X_test` y guárdalas en `figs/`.

---

## 10) **Comparación de tiempos de cómputo**

**Objetivo (entregable):** Comparar **tiempo de entrenamiento y predicción** entre GNB y KDE-NB.

```python
import time, pandas as pd

def medir_tiempos(nombre, modelo, X_tr, y_tr, X_te):
    t0 = time.perf_counter()
    modelo.fit(X_tr.values, y_tr.values)
    t1 = time.perf_counter()
    _ = modelo.predict(X_te.values)
    t2 = time.perf_counter()
    return {"modelo": nombre, "fit_s": t1-t0, "predict_s": t2-t1, "total_s": t2-t0}

rows = []
rows.append(medir_tiempos("GNB", GaussianNB(), X_train, y_train, X_test))
rows.append(medir_tiempos("KDE-NB", KDENaiveBayes(), X_train, y_train, X_test))
tiempos = pd.DataFrame(rows)
tiempos.to_csv("comparativa_tiempos.csv", index=False)
tiempos
```

> **Figura**: diagrama de barras `total_s` por modelo (`figs/tiempos_barras.png`).

---

## 11) **Visualizaciones obligatorias para el informe/presentación**

1. **Distribuciones / “verosimilitud” (por clase):**  
   Para al menos 2 features, superponer **KDE** por clase y (opcional) PDF gaussiana estimada:
   ```python
   import seaborn as sns, matplotlib.pyplot as plt
   feat = "Torque [Nm]"
   plt.figure(figsize=(6,4))
   sns.kdeplot(data=df, x=feat, hue="Machine failure", common_norm=False)
   plt.title(f"KDE por clase — {feat}")
   plt.tight_layout(); plt.savefig("figs/kde_torque.png", dpi=150)
   ```
2. **Curvas ROC y PR** (por modelo):  
   Guárdalas como `figs/roc_gnb.png`, `figs/roc_kde.png`, `figs/pr_gnb.png`, `figs/pr_kde.png`.
3. **Matrices de confusión anotadas** (GNB 0.5, KDE 0.5, KDE óptimo).  
4. **Mapa de calor de correlación** de features.


## 12) Créditos y licencia

- **Dataset:** *AI4I 2020 Predictive Maintenance Dataset* (UCI).  
- **Uso académico.** Puedes adaptar y reutilizar citando este proyecto.

---

