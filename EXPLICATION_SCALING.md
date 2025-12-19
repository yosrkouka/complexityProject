# 📊 Explication du Tableau de Scaling Empirique

## Vue d'ensemble

Ce tableau montre les **temps d'exécution réels** (en secondes) mesurés pour chaque algorithme sur différentes tailles de problème. C'est ce qu'on appelle une **analyse de scaling empirique**.

---

## 📋 Structure du Tableau

Le tableau contient 4 colonnes principales :

| Colonne | Description |
|---------|-------------|
| **size** | Nombre de tâches dans le problème (100, 200, 400, 800) |
| **SA** | Temps d'exécution pour Simulated Annealing (en secondes) |
| **GA** | Temps d'exécution pour Genetic Algorithm (en secondes) |
| **TS** | Temps d'exécution pour Tabu Search (en secondes) |

---

## 📊 Valeurs Exactes

### Simulated Annealing (SA)

| Taille | Runtime (s) | Explication |
|--------|-------------|-------------|
| 100 tâches | **0.3031** | Très rapide pour petit problème |
| 200 tâches | **0.4198** | Légèrement plus lent (×1.38) |
| 400 tâches | **0.7521** | Encore plus lent (×2.48) |
| 800 tâches | **1.4645** | Le plus lent mais toujours rapide (×4.83) |

**Observation** : SA est **le plus rapide** des trois algorithmes. Le temps augmente presque linéairement avec la taille.

---

### Genetic Algorithm (GA)

| Taille | Runtime (s) | Explication |
|--------|-------------|-------------|
| 100 tâches | **4.2552** | Plus lent que SA (×14) |
| 200 tâches | **8.3073** | Presque doublé (×1.95) |
| 400 tâches | **22.8214** | Beaucoup plus lent (×2.75) |
| 800 tâches | **77.4624** | Très lent (×3.39) |

**Observation** : GA est **le plus lent** des trois. Le temps augmente de manière **super-linéaire** (plus que proportionnel à la taille).

---

### Tabu Search (TS)

| Taille | Runtime (s) | Explication |
|--------|-------------|-------------|
| 100 tâches | **3.4346** | Entre SA et GA |
| 200 tâches | **5.9507** | Presque doublé (×1.73) |
| 400 tâches | **10.2054** | Encore doublé (×1.72) |
| 400 tâches | **19.6394** | Presque doublé (×1.92) |

**Observation** : TS est **intermédiaire** en vitesse. Le temps augmente presque linéairement, similaire à SA mais plus lent.

---

## 🔍 Analyse des Résultats

### Comparaison des Performances

**Pour 100 tâches :**
- SA : 0.30s ⚡ (le plus rapide)
- TS : 3.43s (11× plus lent que SA)
- GA : 4.26s (14× plus lent que SA)

**Pour 800 tâches :**
- SA : 1.46s ⚡ (toujours le plus rapide)
- TS : 19.64s (13× plus lent que SA)
- GA : 77.46s (53× plus lent que SA!)

### Facteur de Scaling

**SA** : De 100 à 800 tâches = **×4.83** (presque linéaire)
- 800 tâches = 4.83 × temps de 100 tâches

**TS** : De 100 à 800 tâches = **×5.72** (quasi-linéaire)
- 800 tâches = 5.72 × temps de 100 tâches

**GA** : De 100 à 800 tâches = **×18.21** (super-linéaire!)
- 800 tâches = 18.21 × temps de 100 tâches

---

## 📈 Interprétation

### Pourquoi SA est le plus rapide ?

1. **Algorithme simple** : Une seule solution à évaluer par itération
2. **Pas de population** : Pas besoin de gérer plusieurs solutions
3. **Complexité O(I × n)** : Linéaire avec le nombre de tâches

### Pourquoi GA est le plus lent ?

1. **Population** : Doit évaluer 80 individus par génération
2. **Opérations coûteuses** : Crossover et mutation sur toute la population
3. **Complexité O(G × P × n)** : Multiplie par la taille de la population

### Pourquoi TS est intermédiaire ?

1. **Neighborhood** : Explore 80 voisins par itération
2. **Tabu list** : Gestion mémoire supplémentaire
3. **Complexité O(I × m × n)** : Multiplie par la taille du voisinage

---

## 🎯 Conclusions Pratiques

### Quand utiliser SA ?
- ✅ **Grands problèmes** (1000+ tâches)
- ✅ **Temps limité**
- ✅ **Solution acceptable suffit**

### Quand utiliser GA ?
- ✅ **Petits problèmes** (< 200 tâches)
- ✅ **Temps disponible**
- ✅ **Besoin de meilleure solution**

### Quand utiliser TS ?
- ✅ **Problèmes moyens** (200-500 tâches)
- ✅ **Équilibre vitesse/qualité**
- ✅ **Besoin de diversité**

---

## 📊 Visualisation

Le graphique montre ces valeurs sur une **échelle logarithmique**, ce qui permet de voir clairement :

1. **SA** : Ligne presque droite (scaling linéaire)
2. **TS** : Ligne légèrement courbe (scaling quasi-linéaire)
3. **GA** : Ligne très courbe (scaling super-linéaire)

---

## 💡 Points Clés

1. **SA est toujours le plus rapide**, peu importe la taille
2. **GA devient très lent** sur les grands problèmes
3. **TS offre un bon compromis** vitesse/qualité
4. **Le scaling est différent** pour chaque algorithme
5. **Ces valeurs sont exactes** et proviennent de vos tests empiriques

---

**Ces données confirment l'analyse théorique de complexité et montrent les performances réelles des algorithmes sur votre problème spécifique.**

