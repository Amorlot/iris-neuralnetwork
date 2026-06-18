# Reti Neurali con Keras — Dataset Iris

**Autori:** Andrea Morlotti · Gianluca Cerea

Progetto didattico che applica una rete neurale multi-layer al dataset Iris per la classificazione multi-classe. Include analisi esplorativa (EDA) e confronto tra optimizer.

---

## Struttura del progetto

```
iris-keras/
├── main.py          # Addestramento e valutazione della rete neurale
├── eda.py           # Analisi esplorativa con grafici (EDA)
├── api/
│   └── app.py       # Server Flask con endpoint REST per la predizione
├── test_api.py      # Test degli endpoint REST (richiede il server attivo)
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

## API REST (Flask)

Gli endpoint implementano una **pipeline a stati**: devono essere chiamati nell'ordine indicato. Il server risponde `409 Conflict` se un endpoint viene invocato fuori sequenza.

### Endpoint

| # | Metodo | Endpoint | Descrizione |
|---|---|---|---|
| — | `GET` | `/health` | Stato del server e step corrente |
| — | `GET` | `/status` | Step corrente e prossimo step atteso |
| 1 | `POST` | `/data/load` | Carica e preprocessa il dataset Iris |
| 2 | `POST` | `/model/configure` | Sceglie optimizer, learning rate, epoche |
| 3 | `POST` | `/model/build` | Costruisce l'architettura della rete |
| 4 | `POST` | `/model/train` | Addestra il modello |
| 5 | `POST` | `/predict` | Predice la specie (ripetibile) |

### Ordine obbligatorio degli endpoint

```
POST /data/load
POST /model/configure
POST /model/build
POST /model/train
POST /predict        ← ripetibile quante volte si vuole
```

### Avviare il server

```bash
python api/app.py
```

Il server si avvia su `http://localhost:5000` con stato `idle`.

### Esempio: sequenza completa con curl

```bash
# 1. Carica il dataset
curl -X POST http://localhost:5000/data/load

# 2. Configura l'optimizer (adam o sgd)
curl -X POST http://localhost:5000/model/configure \
     -H "Content-Type: application/json" \
     -d '{"optimizer": "adam", "learning_rate": 0.001, "epochs": 100, "batch_size": 16}'

# 3. Costruisce il modello
curl -X POST http://localhost:5000/model/build

# 4. Addestra (~30s)
curl -X POST http://localhost:5000/model/train

# 5. Predici
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Risposta di `/predict`:

```json
{
  "predicted_class": "setosa",
  "probabilities": {
    "setosa": 0.997412,
    "versicolor": 0.001831,
    "virginica": 0.000757
  }
}
```

### Eseguire i test

Con il server già in esecuzione:

```bash
python test_api.py
```

`test_api.py` percorre l'intera pipeline nell'ordine corretto e verifica anche che le chiamate fuori sequenza restituiscano `409`.

---

## Avvio senza Docker

```bash
pip install -r requirements.txt
python main.py          # training con confronto optimizer
python eda.py           # grafici EDA
python api/app.py       # server REST
python test_api.py      # test degli endpoint (server deve essere attivo)
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
- flask 3.0.3
- requests 2.32.3
