# 📐 Fondements Mathématiques - Architecture CNN-LSTM Tennis

## 1. Features Cinématiques

### Conversion Pixels → Mètres

Soit $(x_p, y_p)$ les coordonnées pixels. La conversion en coordonnées monde $(x_m, y_m)$ utilise la calibration caméra :

$$
\begin{bmatrix} x_m \\ y_m \\ 1 \end{bmatrix} = H^{-1} \begin{bmatrix} x_p \\ y_p \\ 1 \end{bmatrix}
$$

où $H = K [R_{:,0:2} | t]$ avec :
- $K$ : matrice intrinsèque de la caméra
- $R$ : matrice de rotation (rodrigues de rvec)
- $t$ : vecteur de translation

### Dérivées Cinématiques

#### Vitesse
$$
v_x(t) = \frac{dx_m}{dt}, \quad v_y(t) = \frac{dy_m}{dt}
$$

$$
\text{speed}(t) = \sqrt{v_x^2 + v_y^2}
$$

#### Accélération
$$
a_x(t) = \frac{dv_x}{dt}, \quad a_y(t) = \frac{dv_y}{dt}
$$

$$
\text{accel}(t) = \sqrt{a_x^2 + a_y^2}
$$

#### Jerk (Dérivée de l'accélération)
$$
j(t) = \frac{d\text{accel}}{dt} = \sqrt{\left(\frac{da_x}{dt}\right)^2 + \left(\frac{da_y}{dt}\right)^2}
$$

#### Turn Rate (Taux de rotation)
$$
\omega(t) = \frac{v_x \cdot a_y - v_y \cdot a_x}{\text{speed}^2}
$$

---

## 2. Data Windowing

### Fenêtre Centrée

Pour une frame $i$, la fenêtre $W_i$ de taille $w = 2h + 1$ est définie par :

$$
W_i = \{X_{i-h}, X_{i-h+1}, \ldots, X_i, \ldots, X_{i+h-1}, X_{i+h}\}
$$

où :
- $h = \lfloor w/2 \rfloor$ (demi-fenêtre)
- $X_t \in \mathbb{R}^{9}$ (vecteur de features à la frame $t$)

**Exemple** : $w = 31 \Rightarrow h = 15$ → fenêtre de ±15 frames

### Padding

Aux bords, on applique un padding **edge** (réplication) :

$$
X_{t} = 
\begin{cases}
X_0 & \text{si } t < 0 \\
X_t & \text{si } 0 \leq t < T \\
X_{T-1} & \text{si } t \geq T
\end{cases}
$$

### Output

$$
\text{Input CNN-LSTM} : \mathcal{X} \in \mathbb{R}^{B \times w \times F}
$$

où :
- $B$ : batch size
- $w$ : window size (31)
- $F$ : nombre de features (9)

---

## 3. Architecture CNN-LSTM

### Bloc 1D-CNN

#### Convolution 1D

Pour une couche Conv1D avec $C$ filtres de taille $k$ :

$$
y_{t,c} = \sigma\left(\sum_{j=0}^{F-1} \sum_{m=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor} w_{c,j,m} \cdot x_{t+m,j} + b_c\right)
$$

où :
- $w_{c,j,m}$ : poids du filtre $c$ pour la feature $j$ au décalage $m$
- $b_c$ : biais
- $\sigma$ : fonction d'activation (ReLU)

#### Batch Normalization

$$
\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

$$
y = \gamma \hat{x} + \beta
$$

où $\mu_{\mathcal{B}}$ et $\sigma_{\mathcal{B}}^2$ sont la moyenne et variance du batch.

#### MaxPooling1D

$$
y_t = \max_{i \in [t \cdot s, t \cdot s + p)} x_i
$$

avec $p$ : pool size, $s$ : stride

### Bloc Bi-LSTM

#### LSTM Cell (unidirectionnel)

À l'instant $t$, une cellule LSTM calcule :

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) && \text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) && \text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) && \text{(candidate)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t && \text{(cell state)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) && \text{(output gate)} \\
h_t &= o_t \odot \tanh(C_t) && \text{(hidden state)}
\end{aligned}
$$

où :
- $\sigma$ : fonction sigmoid
- $\odot$ : produit élément par élément (Hadamard)
- $W_*, b_*$ : matrices de poids et biais

#### Bi-LSTM

$$
\vec{h}_t = \text{LSTM}_{\text{forward}}(x_t, \vec{h}_{t-1})
$$

$$
\overleftarrow{h}_t = \text{LSTM}_{\text{backward}}(x_t, \overleftarrow{h}_{t+1})
$$

$$
h_t = [\vec{h}_t; \overleftarrow{h}_t]
$$

**Dimension** : Si LSTM a $d$ unités, Bi-LSTM produit $2d$ unités.

#### Global Average Pooling

$$
\text{GAP}(\{h_1, h_2, \ldots, h_T\}) = \frac{1}{T} \sum_{t=1}^{T} h_t
$$

### Classification Head

#### Dense Layer

$$
z = W \cdot x + b
$$

$$
a = \text{ReLU}(z) = \max(0, z)
$$

#### Dropout

Pendant l'entraînement, avec probabilité $p$ :

$$
y_i = 
\begin{cases}
0 & \text{avec proba } p \\
\frac{x_i}{1-p} & \text{avec proba } 1-p
\end{cases}
$$

#### Softmax

$$
P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

où $K = 3$ (air, hit, bounce).

---

## 4. Focal Loss

### Formulation

Pour un exemple $x$ de classe vraie $y$, la Focal Loss est :

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

où :

$$
p_t = 
\begin{cases}
P(y = 1 | x) & \text{si } y = 1 \\
1 - P(y = 1 | x) & \text{sinon}
\end{cases}
$$

### Composantes

#### 1. Cross-Entropy Standard

$$
\text{CE}(p_t) = -\log(p_t)
$$

#### 2. Facteur de Modulation

$$
(1 - p_t)^\gamma
$$

- Si $p_t \to 1$ (exemple facile) : $(1 - p_t)^\gamma \to 0$ → **faible poids**
- Si $p_t \to 0$ (exemple difficile) : $(1 - p_t)^\gamma \to 1$ → **fort poids**

#### 3. Poids Alpha

$$
\alpha_t = 
\begin{cases}
\alpha_0 & \text{si } y = 0 \text{ (air)} \\
\alpha_1 & \text{si } y = 1 \text{ (hit)} \\
\alpha_2 & \text{si } y = 2 \text{ (bounce)}
\end{cases}
$$

**Auto-calcul** : Inversement proportionnel à la fréquence

$$
\alpha_k = \frac{N}{K \cdot n_k}
$$

puis normalisation :

$$
\alpha_k \leftarrow \frac{\alpha_k}{\sum_{j=0}^{K-1} \alpha_j}
$$

### Extension Multi-classe

Pour $K$ classes :

$$
\text{FL} = -\sum_{k=0}^{K-1} \alpha_k (1 - p_k)^\gamma y_k \log(p_k)
$$

où $y_k \in \{0, 1\}$ (one-hot encoding).

---

## 5. Métriques

### F1-Score

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

où :

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

### F1-Score Macro

$$
F1_{\text{macro}} = \frac{1}{K} \sum_{k=0}^{K-1} F1_k
$$

**Important** : Donne le même poids à chaque classe (crucial pour classes rares).

### Precision-Recall AUC

Pour une classe $k$ :

1. Calculer les paires $(R_i, P_i)$ pour différents seuils $\theta_i$ :

$$
R_i = \frac{\text{TP}(\theta_i)}{\text{TP}(\theta_i) + \text{FN}(\theta_i)}
$$

$$
P_i = \frac{\text{TP}(\theta_i)}{\text{TP}(\theta_i) + \text{FP}(\theta_i)}
$$

2. Calculer l'aire sous la courbe PR :

$$
\text{PR-AUC} = \int_0^1 P(R) \, dR
$$

Approximation numérique (trapèzes) :

$$
\text{PR-AUC} \approx \sum_{i=1}^{N-1} \frac{1}{2}(P_i + P_{i+1})(R_{i+1} - R_i)
$$

---

## 6. Post-Processing (NMS)

### Recherche de Pics

Pour une série temporelle de probabilités $p(t)$ :

**Condition de pic** à $t$ :

$$
\begin{cases}
p(t) > \theta & \text{(seuil de confiance)} \\
p(t) > p(t-1) & \text{(croissant avant)} \\
p(t) > p(t+1) & \text{(décroissant après)} \\
|t - t_j| \geq d_{\min} & \forall j \text{ (distance min avec autres pics)}
\end{cases}
$$

### Suppression Non-Maxima

1. Trier les pics par confiance décroissante : $\{t_1, t_2, \ldots, t_M\}$

2. Pour chaque pic $t_i$ :
   - Si $\exists t_j$ gardé tel que $|t_i - t_j| < d_{\min}$ → **supprimer** $t_i$
   - Sinon → **garder** $t_i$

### Résultat

Ensemble des frames détectées :

$$
\mathcal{D} = \{(t_i, p(t_i)) \mid t_i \text{ pic conservé}\}
$$

---

## 7. Complexité Computationnelle

### Forward Pass

#### CNN
- Conv1D(64, k=5) : $O(B \cdot T \cdot F \cdot C \cdot k) = O(B \cdot 31 \cdot 9 \cdot 64 \cdot 5)$
- Conv1D(128, k=3) : $O(B \cdot T \cdot C_1 \cdot C_2 \cdot k) = O(B \cdot 31 \cdot 64 \cdot 128 \cdot 3)$

#### Bi-LSTM
- LSTM(128) : $O(B \cdot T \cdot d \cdot (d + C))$ où $d = 128$

**Total** : $O(B \cdot T \cdot C^2)$ avec $C \sim 256$

### Entraînement

Pour $E$ epochs, $N$ samples :

$$
\text{Temps total} \propto E \cdot \frac{N}{B} \cdot O(\text{forward + backward})
$$

**Estimation** :
- $N = 100\,000$ fenêtres
- $B = 256$
- $E = 50$ epochs
- GPU (RTX 3080) : ~2-3 heures

---

## 8. Stabilité Numérique

### Clipping pour Softmax

Avant le log :

$$
p_k \leftarrow \text{clip}(p_k, \epsilon, 1 - \epsilon)
$$

avec $\epsilon = 10^{-7}$

### Gradient Clipping

$$
\nabla \leftarrow 
\begin{cases}
\nabla & \text{si } \|\nabla\| \leq \tau \\
\frac{\tau}{\|\nabla\|} \nabla & \text{sinon}
\end{cases}
$$

Typiquement $\tau = 5.0$.

---

## 9. Hyperparamètres Optimaux

| Hyperparamètre | Valeur | Justification |
|----------------|--------|---------------|
| **Window Size** | 31 | ±15 frames = 300ms à 50 FPS |
| **CNN Filters** | [64, 128, 256] | Augmentation progressive |
| **LSTM Units** | [128, 64] | Capacité suffisante sans overfitting |
| **Dropout** | [0.3, 0.4] | Régularisation |
| **L2 Regularization** | 0.001 | Poids pénalisés |
| **Focal γ** | 2.0 | Standard pour déséquilibre extrême |
| **Learning Rate** | 0.001 | Adam par défaut |
| **Batch Size** | 256 | Compromise vitesse/stabilité |

---

## 10. Limitations & Améliorations

### Limitations Actuelles

1. **Fenêtre fixe** : Ne s'adapte pas à la vitesse de la balle
2. **Features manuelles** : Pas d'apprentissage end-to-end depuis pixels
3. **Séquence unique** : Pas de mémoire long terme (>31 frames)

### Améliorations Futures

#### 1. Attention Mechanism

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Permet de **focaliser** sur les frames importantes.

#### 2. Temporal Convolutional Network (TCN)

$$
y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t - 2^l \cdot i}
$$

Récepteur field exponentiel : $2^L \cdot k$

#### 3. Multi-Scale Features

Combiner des fenêtres de tailles différentes :

$$
\mathcal{F} = \text{Concat}(W_{15}, W_{31}, W_{63})
$$

---

**Références Mathématiques** :
- Hochreiter & Schmidhuber (1997) : LSTM
- Lin et al. (2017) : Focal Loss
- Vaswani et al. (2017) : Attention Mechanism
- Lea et al. (2017) : Temporal Convolutional Networks
