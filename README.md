# Reti Neurali con Keras — Dataset Iris

**Autori:** Andrea Morlotti · Gianluca Cerea

Progetto didattico che applica una rete neurale multi-layer al dataset Iris per la classificazione multi-classe. Include analisi esplorativa (EDA) e confronto tra optimizer.

---

## Struttura del progetto

```
iris-keras/
├── main.py          # Addestramento e valutazione della rete neurale
├── eda.py           # Analisi esplorativa con grafici (EDA)
├── Dockerfile       # Container per eseguire il progetto
└── requirements.txt # Dipendenze Python
```

---

## Dataset

Il dataset **Iris** (Fisher, 1936) contiene 150 campioni di fiori suddivisi in 3 specie:

| Specie | Campioni |
|---|---|
| Iris Setosa | 50 |
| Iris Versicolor | 50 |
| Iris Virginica | 50 |

**Feature (in cm):** `sepal length`, `sepal width`, `petal length`, `petal width`

**Obiettivo:** classificare la specie in base alle 4 misurazioni.

---

## Architettura della rete

```
Input(4) → Dense(16, ReLU) → Dense(16, ReLU) → Dense(3, Softmax)
```

| Componente | Scelta | Motivazione |
|---|---|---|
| Attivazione hidden | ReLU | Veloce, risolve il vanishing gradient |
| Attivazione output | Softmax | Produce probabilità per 3 classi |
| Loss | categorical_crossentropy | Standard per multi-classe con one-hot |
| Optimizer | SGD (lr=0.01) | Testato anche con Adam |
| Parametri totali | 403 | 4×16+16 + 16×16+16 + 16×3+3 |

**Training:** 100 epoche, batch size 16, validation split 10%

---

## Risultati

| Optimizer | Test Accuracy |
|---|---|
| SGD (lr=0.01) | ~96-97% |
| Adam (lr=0.001) | ~93-97% |

---

## Avvio con Docker

```bash
# Build
docker build -t iris-keras .

# Esegui il training (main.py)
docker run --rm iris-keras

# Esegui l'EDA — genera i grafici PNG in ./output
mkdir -p output
docker run --rm -v $(pwd)/output:/app/output iris-keras python eda.py
```

### Grafici EDA generati

| File | Contenuto |
|---|---|
| `plot_scatter_petal.png` | Scatter petal_length vs petal_width per specie |
| `plot_pairplot.png` | Tutte le combinazioni di feature |
| `plot_boxplot.png` | Distribuzione per feature e specie |
| `plot_kde.png` | Densità KDE per feature e specie |
| `plot_heatmap.png` | Matrice di correlazione |

---

## Avvio senza Docker

```bash
pip install -r requirements.txt
python main.py   # training
python eda.py    # grafici EDA
```

---

## Dipendenze

- Python 3.11
- TensorFlow 2.16.1
- scikit-learn 1.4.2
- NumPy 1.26.4
- pandas 2.2.2
- matplotlib 3.9.0
- seaborn 0.13.2
