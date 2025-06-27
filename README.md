# SinGAN – Single‑Image Generative Adversarial Networks

Bienvenue ! Ce dépôt contient notre ré‑implémentation de **SinGAN** et plusieurs variantes explorées dans le cadre du projet *4IM06*. Le rapport complet se trouve dans **`Paper/Rapport.pdf`**.

---

## Arborescence générale

| Chemin / Fichier               | Rôle                                                  | Commentaires clés                                                                                                                                                                                                                                                                                                  |
| ------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`Paper/Rapport.pdf`**        | Rapport de projet                                     | Toutes les explications théoriques et résultats expérimentaux.                                                                                                                                                                                                                                                     |
| **`singan.ipynb`**                  | Implémentation *fidèle* du papier **SinGAN** original | Réseau pyramidal multi‑échelle ; un générateur + discriminateur par niveau.                                                                                                                                                                                                                                        |
| **`train_SW_feature.ipynb`**      | Variante **sans discriminateur**                      | Remplace la loss adversariale par une distance Sliced Wasserstein (SWD) calculée sur des features Inception v3. Détaillée section 5 du rapport.                                                                                                                                                                    |
| **`utils_ot.py`**              | Outils transport optimal                              | Fonctions de calcul des distances Wasserstein / SWD utilisées par `train_SW_feature.py`.                                                                                                                                                                                                                           |
| **`OT_verif.py`** ou notebook  | Script de validation                                  | Vérifie que la distance de transport est faible entre images similaires et élevée entre images dissemblables. Principalement pour debug / sanity‑check.                                                                                                                                                            |
| **`metrics.ipynb`**                  | Calculs des différentes métriques | Utiliation de Wasserstein, SWD et LPIPS pour l'évaluation quantitative des images générées                                                                                                                                                                                                                                        |
| **`models/`**                  | Poids & hyper‑paramètres                              | Organisation par **image source** :<br>  • `basic/` → réseaux issus notre implémentation du  papier ;<br>  • `swd/`  → réseaux issus de la variante SWD ;<br>  • partagé. Contient également les bruit(s) injectés à chaque échelle. |
| **`outputs/`** (ou **`50/`**)  | Exemples d’images générées                            | Dossiers triés par image‑source et par méthode (basic / swd ). Utile pour comparer qualitativement les modèles pré‑entraînés.                                                                                                                                                                             |

---

## Utilisation rapide

1. **Installer les dépendances** listées dans `requirements.txt` :

   ```bash
   pip install -r requirements.txt
    ```
2. **Utiliser des poids déjà entraînés** 
Lancez `use_model.ipynb` en précisant l’image d’entrée et le modèle voulu.

3. **Ré-entraîner sur une nouvelle image** 
Ouvrez et exécutez le notebook `singan.ipyn` ou `train_SW_feature.ipyn`.


> Remarque : Dans le fichier `to_one_model.ipynb` vous trouverez un début de test d'architecture avec un unique générateur et discriminateur pour essayer d'éviter les redondances.
