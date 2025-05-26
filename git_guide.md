# Workflow Git pour la Création d'une Branche et l'Intégration des Modifications

**Vocabulaire** : <br>
> `origin` c’est l’url du dépôt par défaut dans git <br>
> `main` c'est la branch de base par défaut sur git

```bash
Git checkout main
````
-> décale sur le main

```bash
Git pull origin main`
```
-> pull en local de la branch main.


## 1. Création de la Branche (ici feature)

Créez et poussez votre branche dès le début pour que tout le monde puisse la voir et pour régler d'éventuels problèmes de permissions Git dès le départ.

```bash
git checkout -b feature
git push -u origin feature
```

> Le flag `-u` (ou `--set-upstream`) permet d'établir la branche distante par défaut. Ainsi, prochains `push` et `pull` vont se faire automatiquement sur cette branche sans avoir besoin de préciser `origin feature`.

## 2. Développement et Commits

Effectuez vos modifications et enregistrez-les avec des commits descriptifs. 
Faites de push régulier pour que tout le monde puisse voir ce que vous faites.

```bash
git add <vos-fichiers>
git commit -m "Message descriptif de vos modifications"
```

## 3. Synchronisation avec la Branche Main et Résolution des Conflits

### a. Récupérer les Dernières Modifications

Avant de fusionner vos changements sur `main`, récupérez les dernières mises à jour du dépôt distant :

```bash
git fetch origin
```

### b. Rebase sur `origin/main`

- **Si personne n'a modifié votre branche :**
  ```bash
  git rebase origin/main
  ```
  Le rebase rejoue vos commits au-dessus de `origin/main`, conservant ainsi un historique linéaire et propre.

- **Si quelqu'un a déjà travaillé sur votre branche :**
  1. Effectuez un rebase sur votre branche distante (si nécessaire) :
     ```bash
     git rebase origin/ma-branche
     ```
  2. Résolvez les conflits qui peuvent survenir.
  3. Puis, refaites un rebase sur `origin/main` :
     ```bash
     git rebase origin/main
     ```
  4. Résolvez à nouveau les conflits si nécessaire.

### c. Résolution des Conflits

Lorsqu'un conflit survient pendant le rebase, Git vous indique les fichiers concernés avec des marqueurs :

```diff
<<<<<<< HEAD
// Code provenant de origin/main
=======
 // Vos modifications
>>>>>>> Votre commit
```

Pour résoudre un conflit :
1. Ouvrez le fichier concerné dans l'éditeur.
2. Choisissez quelle partie du code garder (ou fusionnez les deux).
3. Supprimez les marqueurs de conflit.
4. Ajoutez le fichier résolu :
   ```bash
   git add <fichier_concerné>
   ```
5. Continuez le rebase :
   ```bash
   git rebase --continue
   ```

## 4. Tester le Code

Avant de pousser vos modifications, vérifiez que tout fonctionne correctement :

```bash
python3 launcher.py
```

## 5. Force Push

Après un rebase, l'historique de votre branche a été réécrit. Pour mettre à jour la branche distante, forcez le push :

```bash
git push -f
```

## 6. Création de la Pull Request (PR)

1. **Accédez à GitHub  :**  
   Rendez-vous sur votre dépôt.

2. **Créer une nouvelle PR :**  
   - Sélectionnez comme base la branche `Loudwig/SinGan/main`.
   - Sélectionnez comme branche de comparaison votre branche (`feature`).

3. **Soumettre la PR :**  
   Ajoutez un titre et une description détaillée de vos modifications. La PR pourra être examinée par l'équipe, et une fois validée, elle sera fusionnée.

## 7. Delete votre branche en local et sur le repo

1. **En Local :**
    ```bash
    git branch -d feature         
    ```
2. **Sur le repo :**
    ```bash
    git push origin --delete feature
    ```
