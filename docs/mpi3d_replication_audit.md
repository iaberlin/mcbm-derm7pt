# MPI3D-Replikationsaudit

**Stand:** 30.09.2026\
**Projekt:** Masterarbeit -- MCBM / Derm7pt\
**Gegenstand:** Replikationsaudit der MPI3D-MCBM-Pipeline nach Almudevar
et al.

## 1. Ziel und Motivation

Aufgrund der deutlichen Abweichung der eigenen MPI3D-MCBM Task Accuracy
von ca. 64 % gegenüber den von Almudevar et al. berichteten 92.7 ± 1.1 %
wurde zunächst die Daten- und Trainingspipeline kontrolliert, bevor
weitere MCBM-Varianten oder Hyperparameter untersucht werden.

Dieses Vorgehen folgt dem für die Masterarbeit festgelegten
diagnostischen Ablauf: Zunächst sollen insbesondere Splits,
Preprocessing, Label-Mapping, Filterung und weitere
Implementierungsdetails geprüft und das Experimentalprotokoll
eingefroren werden. Erst anschließend sollen kontrollierte Baselines zur
Lokalisierung der Abweichung eingesetzt werden.

## 2. Eingefrorener Replikationsstand

Der für die MPI3D-Replikation verwendete Stand wurde in Git eingefroren:

``` text
Tag:    mpi3d-replication-freeze-v1
Commit: 7b94303c038618eb752b926f24cedb8a9ea77c78
```

MPI3D-relevante Änderungen wurden dabei von parallel vorhandenen
CUB/Derm7pt-Arbeiten getrennt.

Der Trainingspfad verwendet `bin/train.py`. Der dort übergebene CLI-Seed
wird an `TrainExperiment` weitergegeben und in `BaseExperiment` als
`self.seed` gespeichert.

## 3. MPI3D-Datensatz und Faktorstruktur

Die verwendete Datei `mpi3d_real.npz` enthält `images.npy` mit:

``` text
shape: (1,036,800, 64, 64, 3)
dtype: uint8
```

Die Samplezahl entspricht exakt:

\[ 6 `\cdot 6`{=tex} `\cdot 2`{=tex} `\cdot 3`{=tex} `\cdot 3`{=tex}
`\cdot 40`{=tex} `\cdot 40`{=tex} = 1{,}036{,}800. \]

Im MPI3D-Loader werden sieben Faktoren mit folgenden Kardinalitäten
verwendet:

``` text
object_color       6
object_shape       6
object_size        2
camera_height      3
background_color   3
horizontal_axis   40
vertical_axis     40
```

Für das Replikationsexperiment gilt:

-   Task (y): `object_shape`
-   Concepts (c): `object_color`, `object_size`, `camera_height`,
    `background_color`
-   task-related nuisance (n_y): `horizontal_axis`
-   task-unrelated nuisance (n\_{`\bar `{=tex}y}): `vertical_axis`

Diese Rollen entsprechen der im Paper beschriebenen MPI3D-Konstruktion.

## 4. Adapter für das verwendete MPI3D-Dateiformat

Der Commit

``` text
48b1dfa  prep for mcbm replica on mpi3d
```

passte den ursprünglichen MPI3D-Loader an.

Der ursprüngliche Code erwartete:

``` python
self.images = dataset['images']
self.labels = dataset['labels']
```

Die verwendete MPI3D-Datei enthält dagegen Bilder, aber keine separat
gespeicherten Labels. Der Adapter verwendet daher:

``` python
self.images = dataset['images'].transpose(0, 3, 1, 2)

if "labels" in dataset.files:
    self.labels = dataset["labels"]
else:
    factor_sizes = [6, 6, 2, 3, 3, 40, 40]
    self.labels = np.indices(factor_sizes).reshape(7, -1).T
```

Die Transposition transformiert die vorliegende Bildrepräsentation von
NHWC `(N, 64, 64, 3)` nach NCHW.

### 4.1 Empirische Prüfung der Label-Rekonstruktion

Da die Rekonstruktion der Faktorlabels eine korrekte Reihenfolge der
Bilder voraussetzt, wurde diese Annahme separat geprüft.

Dazu wurde ein Audit-Skript erstellt, das bei konstant gehaltenen
übrigen Faktoren jeweils genau einen der sieben Faktoren verändert:

-   `object_color`
-   `object_shape`
-   `object_size`
-   `camera_height`
-   `background_color`
-   `horizontal_axis`
-   `vertical_axis`

Für alle sieben Dimensionen entsprach die sichtbare Änderung dem jeweils
rekonstruierten Faktor. Insbesondere zeigten sich die erwarteten
Änderungen von Objektfarbe, Objektform, Objektgröße, Kamerahöhe,
Hintergrundfarbe sowie horizontaler und vertikaler Objektposition.

Die Indexdifferenzen waren ebenfalls mit der angenommenen
C-order-Faktorstruktur konsistent. Beispielsweise änderte sich der Index
bei einer Änderung der `vertical_axis` um 10 um 10 und bei einer
Änderung der `horizontal_axis` um 10 um 400.

**Bewertung:** Die angenommene Faktorordnung und das
Image↔Factor-Alignment sind für kontrollierte Einzel-Faktor-Änderungen
über alle sieben Faktoren empirisch verifiziert. Es wurde keine
exhaustive Prüfung aller 1.036.800 Bild-Label-Paare durchgeführt.

Die zugehörige Visualisierung wird als
`helper_replica/mpi3d_factor_order_audit.png` gespeichert.

## 5. MCBM-Filterung

Die Filterimplementierung erzeugt zunächst sämtliche Kombinationen aus
Concepts und `horizontal_axis`:

\[ 6 `\cdot 2`{=tex} `\cdot 3`{=tex} `\cdot 3`{=tex} `\cdot 40`{=tex} =
4{,}320. \]

Diese Kombinationen werden zufällig gemischt und gleichmäßig den sechs
Task-Klassen zugeordnet. Dadurch soll jede Kombination ((c,n_y)) genau
einem Task-Label (y) entsprechen.

Der unabhängige Audit ergab:

``` text
=== FILTER ===
(c, n_y) combinations: 4320
Allowed (y, c, n_y) combinations: 4320
Samples after filter: 172800

=== UNIQUENESS CHECK ===
Unique (c, n_y): 4320
Violations of (c, n_y) -> y: 0
```

Da `vertical_axis` nicht in die Filterbedingung eingeht, bleiben für
jede erlaubte ((y,c,n_y))-Kombination 40 Werte erhalten:

\[ 4{,}320 `\cdot 40`{=tex} = 172{,}800. \]

Da die 4.320 Kombinationen gleichmäßig auf sechs Task-Klassen verteilt
werden, enthält jede Shape-Klasse vor dem Train/Test-Split exakt:

\[ 720 `\cdot 40`{=tex} = 28{,}800 \]

Samples.

**Bewertung:** Filterkonstruktion und Eindeutigkeitsbedingung
((c,n_y)`\rightarrow `{=tex}y) verifiziert.

## 6. Train/Test-Split

Der gefilterte Datensatz wird zufällig permutiert und anschließend im
Verhältnis 75/25 aufgeteilt.

Der Audit ergab:

``` text
=== SPLIT ===
Train: 129600
Test: 43200
Train/Test overlap: 0
```

Die Task-Verteilungen waren:

``` text
train
  shape 0: 21489 (0.1658)
  shape 1: 21589 (0.1666)
  shape 2: 21565 (0.1664)
  shape 3: 21601 (0.1667)
  shape 4: 21638 (0.1670)
  shape 5: 21718 (0.1676)

test
  shape 0: 7311 (0.1692)
  shape 1: 7211 (0.1669)
  shape 2: 7235 (0.1675)
  shape 3: 7199 (0.1666)
  shape 4: 7162 (0.1658)
  shape 5: 7082 (0.1639)
```

Für jede Klasse summieren sich Train- und Test-Samples auf exakt 28.800.
Die leichten Abweichungen innerhalb der beiden Splits entstehen daher
durch die zufällige Sample-Aufteilung und nicht durch eine
Klassenimbalance des gefilterten Gesamtdatensatzes.

**Bewertung:** 75/25-Split und Sample-Disjunktheit verifiziert.

## 7. Seed-Propagation für Dataset und Split

`bin/train.py` definiert `--seed` als Integer mit Defaultwert 42 und
reicht die Argumente an `TrainExperiment` weiter. `BaseExperiment`
speichert diesen Wert als `self.seed`.

`TrainExperiment` erzeugt anschließend Train- und Test-Loader mit
demselben Seed:

``` python
self.train_loader, self.model_kwargs, _ = get_loader(
    train=True, seed=self.seed, ...)

self.test_loader, _, _ = get_loader(
    train=False, seed=self.seed, ...)
```

Innerhalb von `DisentanglementDataset._filter_samples_nuisances()` wird
vor der Filterkonstruktion

``` python
random.seed(self.seed)
```

gesetzt. Anschließend erfolgen `random.shuffle(combinations)` und in
`_select_samples()` `random.sample(...)`.

Da Train- und Test-Dataset separat mit demselben Seed konstruiert
werden, reproduzieren beide Instanzen dieselbe Filterzuordnung und
dieselbe Sample-Permutation. Die Train-Instanz verwendet die ersten 75
%, die Test-Instanz die verbleibenden 25 %.

**Bewertung:** Dataset-Seed, Filter und Train/Test-Split sind
nachvollziehbar und konsistent.

## 8. Trainingskonfiguration

Die geprüfte MPI3D-MCBM-Replikationskonfiguration verwendet:

``` text
encoder          ResNet20
gamma            1
epochs           50
batch size       128
optimizer        SGD
base LR          0.003
base batch size  64
momentum         0.9
weight decay     4e-5
scheduler        StepLR
step size        20
scheduler gamma  0.1
```

Die Funktion `get_optimizer()` berechnet die effektive Learning Rate
als:

``` python
lr = base_lr * batch_size / base_batch_size
```

Damit ergibt sich:

\[ 0.003 `\cdot `{=tex}`\frac{128}{64}`{=tex} = 0.006. \]

Die effektive Learning Rate beträgt somit 0.006. ResNet20, 50 Epochen,
Batch Size 128, SGD, Momentum 0.9, Weight Decay
(4`\times10`{=tex}\^{-5}), StepLR mit Step Size 20 und Scheduler-Gamma
0.1 entsprechen den für MPI3D dokumentierten Einstellungen bei Almudevar
et al.

**Bewertung:** Die bislang geprüften zentralen Optimizer-, Scheduler-
und Architekturparameter sind mit der dokumentierten MPI3D-Konfiguration
konsistent.

## 9. Verbleibende Reproduzierbarkeitsbeobachtung: globales Seeding

Der CLI-Seed wird nachweislich für die Dataset-Konstruktion verwendet.
Eine Code-Suche ergab im normalen Trainingspfad jedoch bislang keine
allgemeine globale Initialisierung des PyTorch-RNG über beispielsweise:

``` python
torch.manual_seed(self.seed)
```

Die gefundenen Seed-Aufrufe betreffen insbesondere den Python-RNG
innerhalb verschiedener Dataset-Klassen, spezielle
Disentanglement-Hilfsfunktionen sowie einzelne andere Komponenten.

Damit kontrolliert der Experiment-Seed nach dem derzeitigen Audit
nachweislich die MPI3D-Dataset-Konstruktion, aber nicht nachweislich
sämtliche Zufallsquellen der Modellinitialisierung und des
GPU-Trainings.

Dies ist als Reproduzierbarkeitsaspekt festzuhalten. Aus dem Audit folgt
jedoch nicht, dass dieser Punkt die beobachtete große
Task-Accuracy-Differenz verursacht.

Der Originalcode wird deshalb an dieser Stelle für die Replikation
zunächst nicht verändert.

## 10. Zusammenfassung des Audits

  Prüfpunkt                              Ergebnis
  -------------------------------------- ------------------------------------------
  MPI3D Samplezahl                       ✓ konsistent
  Bilddimensionen / dtype                ✓ konsistent
  Faktor-Kardinalitäten                  ✓ konsistent
  Faktorordnung                          ✓ empirisch geprüft
  Image↔Factor-Alignment                 ✓ kontrollierte Einzel-Faktor-Änderungen
  Rollen (y,c,n_y,n\_{`\bar `{=tex}y})   ✓ konsistent mit Paper
  MCBM-Filter                            ✓ 172.800 Samples
  ((c,n_y)`\rightarrow `{=tex}y)         ✓ 0 Verletzungen
  Klassenverteilung vor Split            ✓ exakt balanciert
  Train/Test-Split                       ✓ 129.600 / 43.200
  Train/Test Sample-Overlap              ✓ 0
  Dataset-Seed-Propagation               ✓ nachvollzogen
  Optimizer                              ✓ SGD
  effektive Learning Rate                ✓ 0.006
  Momentum                               ✓ 0.9
  Weight Decay                           ✓ (4`\times10`{=tex}\^{-5})
  Scheduler                              ✓ StepLR, Step Size 20, Gamma 0.1
  globale PyTorch-Seeds                  ⚠ nicht nachgewiesen

## 11. Fazit und Entscheidung für den nächsten Schritt

Der Audit hat bislang keinen offensichtlichen Fehler in
MPI3D-Datenstruktur, Faktorzuordnung, Labelrekonstruktion,
Filterbedingung, Train/Test-Split oder den zentralen geprüften
Trainingsparametern identifiziert.

Die beobachtete Abweichung der eigenen MCBM Task Accuracy von ca. 64 %
gegenüber den von Almudevar et al. berichteten 92.7 ± 1.1 % kann anhand
dieser Prüfungen daher nicht auf einen dieser Punkte zurückgeführt
werden.

Entsprechend dem für die Masterarbeit festgelegten diagnostischen
Vorgehen wird die Fehlersuche an dieser Stelle nicht unmittelbar durch
einen detaillierten MCBM-Code-Audit ausgeweitet. Stattdessen folgt als
nächster kontrollierter Schritt eine Vanilla-Baseline mit identischer
Datenpipeline und identischem Split. Ziel ist es, zunächst zu bestimmen,
ob die Abweichung bereits auf Ebene von Daten, Labels oder Training
auftritt oder erst durch CBM-/MCBM-spezifische Komponenten entsteht.

Dabei ist zwischen zwei Zielen zu unterscheiden:

1.  Für die strikte MPI3D-Replikation von Almudevar et al. ist die im
    Paper dokumentierte ResNet20-Konfiguration relevant.
2.  Für den diagnostischen Arbeitsplan der Masterarbeit ist die separat
    vorgesehene kontrollierte Vanilla-Baseline maßgeblich.


