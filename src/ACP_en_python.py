import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

                                        
X = np.array([
    [18, 0.5, 0.1, 6.7, 0.5, 2.1, 2, 0, 26.4, 41.5, 2.1],
    [14.1, 0.8, 0.1, 15.3, 1.9, 3.7, 0.5, 0, 29.8, 31.3, 2.5],
    [13.6, 0.7, 0.7, 6.8, 0.6, 7.1, 0.7, 0, 33.8, 34.4, 1.7],
    [14.3, 1.7, 1.7, 6.9, 1.2, 7.4, 0.8, 0, 37.7, 26.2, 2.2],
    [10.3, 1.5, 0.4, 9.3, 0.6, 8.5, 0.9, 0, 38.4, 27.2, 3],
    [13.4, 1.4, 0.5, 8.1, 0.7, 8.6, 1.8, 0, 38.5, 25.3, 1.9],
    [13.5, 1.1, 0.5, 9, 0.6, 9, 3.4, 0, 36.8, 23.5, 2.6],
    [12.9, 1.4, 0.3, 9.4, 0.6, 9.3, 4.3, 0, 41.1, 19.4, 1.3],
    [12.3, 0.3, 0.1, 11.9, 2.4, 3.7, 1.7, 1.9, 42.4, 23.1, 0.2],
    [7.6, 1.2, 3.2, 5.1, 0.6, 5.6, 1.8, 10, 29, 35, 0.9],
    [10.5, 0.3, 0.4, 4.5, 1.8, 6.6, 2.1, 10.1, 19.9, 41.6, 2.3],
    [10, 0.6, 0.6, 9, 1, 8.1, 3.2, 11.8, 28, 25.8, 2],
    [10.6, 0.8, 0.3, 8.9, 3, 10, 6.4, 13.4, 27.4, 19.2, 0],
    [8.8, 2.6, 1.4, 7.8, 1.4, 12.4, 6.2, 11.3, 29.3, 18.5, 0.4],
    [10.1, 1.1, 1.2, 5.9, 1.4, 9.5, 6, 5.9, 40.7, 18.2, 0],
    [15.6, 1.6, 10, 11.4, 7.6, 8.8, 4.8, 3.4, 32.2, 4.6, 0],
    [11.2, 1.3, 16.5, 12.4, 15.8, 8.1, 4.9, 3.4, 20.7, 4.2, 1.5],
    [12.9, 1.5, 7, 7.9, 12.1, 8.1, 5.3, 3.9, 36.1, 5.2, 0],
    [10.9, 5.3, 9.7, 7.6, 9.6, 9.4, 8.5, 4.6, 28.2, 6.2, 0],
    [13.1, 4.4, 7.3, 5.7, 9.8, 12.5, 8, 5, 26.7, 7.5, 0],
    [12.8, 4.7, 7.5, 6.6, 6.8, 15.7, 9.7, 5.3, 24.5, 6.4, 0.1],
    [12.4, 4.3, 8.4, 9.1, 6, 19.5, 10.6, 4.7, 19.8, 3.5, 1.8],
    [11.4, 6, 9.5, 5.9, 5, 21.1, 10.7, 4.2, 20, 4.4, 1.9],
    [12.8, 2.8, 7.1, 8.5, 4, 23.8, 11.3, 3.7, 18.8, 7.2, 0]
])

# Noms des variables de l'échantillon X

noms_variables = ['PVP', 'AGR', 'CMI', 'TRA', 'LOG', 'EDU', 'ACS', 'ACO', 'DEF', 'DET', 'DIV']

# Tableau des stats élémentaires

X_centered = np.round(np.mean(X, axis=0), 6)
X_std = np.round(np.std(X, axis=0), 6)
X_etendues = np.round(np.ptp(X, axis=0), 6)  # Range (max - min)
X_q25 = np.round(np.percentile(X, 25, axis=0), 6)
X_q75 = np.round(np.percentile(X, 75, axis=0), 6)

data = {
    'Variable': noms_variables,
    'Moyenne': X_centered,
    'Écart type': X_std,
    'Étendue': X_etendues,
    'Q25': X_q25,
    'Q75': X_q75
}

print("Données initiales")
df = pd.DataFrame(data)
separator_line = '-' * 50
separator_row = pd.Series(['-' for _ in range(df.shape[1])], index=df.columns)
print(df.to_string(index=False))
print(separator_line)
# Trie des variables en fonction de l'écart type (de la plus grande à la plus petite)

print("Données initiales triées en fonction de leur écart type")
sorted_variables_std_deviation = df.sort_values(by='Écart type', ascending=False)
print(sorted_variables_std_deviation)

# Diviser les variables en deux groupes : celles qui contribuent le plus et le moins  seuil = 4
print("Division des variables en 2 groupes : celles qui cotisent le plus et le moins à l'écart type ")
threshold_std_deviation = 4
top_contributors_std_deviation = \
sorted_variables_std_deviation[sorted_variables_std_deviation['Écart type'] >= threshold_std_deviation][
    'Variable'].tolist()
low_contributors_std_deviation = \
sorted_variables_std_deviation[sorted_variables_std_deviation['Écart type'] < threshold_std_deviation][
    'Variable'].tolist()

# Afficher les résultats
print("Variables qui contribuent le plus à l'écart type:", top_contributors_std_deviation)
print("Variables qui contribuent le moins à l'écart type:", low_contributors_std_deviation)

#######################################################################################################################################################

                                # 2ème partie : Calculs des matrices de Covariances et Corrélations

#Matrices de Cov, Corr & Std
X_centered = X - np.mean(X, axis=0)
X_std = np.std(X_centered, axis=0)
X_standardise = X_centered / X_std
X_corrcoef = np.corrcoef(X_standardise, rowvar=False)


#Graphiques de Cov et Corr

df = pd.DataFrame(X, columns=noms_variables)
Covariance = np.cov(df, rowvar=False)
Correlation = np.corrcoef(df, rowvar=False)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Graphique  matrice de covariance
axes[0].matshow(Covariance, cmap="coolwarm")
axes[0].set_xticks(range(-1, len(noms_variables)))
axes[0].set_xticklabels([''] + noms_variables, rotation=45, ha='right')
axes[0].set_yticks(range(len(noms_variables)))
axes[0].set_yticklabels(noms_variables)
axes[0].set_title("Matrice de covariance")
#
cax = axes[1].matshow(Correlation, cmap="coolwarm")
axes[1].set_xticks(range(-1, len(noms_variables)))
axes[1].set_xticklabels([''] + noms_variables, rotation=45, ha='right')
axes[1].set_yticks(range(len(noms_variables)))
axes[1].set_yticklabels(noms_variables)
axes[1].set_title("Matrice de corrélation")
fig.colorbar(cax, ax=axes[1])
plt.show()

################################################################################################################################################"

                                # VECTEUR PROPRES ET VALEURS PROPRES

valeurs_propres, vecteurs_propres = np.linalg.eig(X_corrcoef)
print(" Vecteurs propres")
print(vecteurs_propres)
#On trie les vecteurs propres selon les valeurs propres triées dans l'ordre décroissant
indices_tries = np.argsort(valeurs_propres)[::-1]
valeurs_propres = valeurs_propres[indices_tries]
vecteurs_propres = vecteurs_propres[:, indices_tries]
eigenvalues = [float("{:.8f}".format(v)) for v in valeurs_propres]
print("")
print("Voici nos valeurs propres ordonnées dans l'ordre décroissant, et il y en" f" {len(eigenvalues)}  au total")
print(eigenvalues)
#############################################################################################################################

                                # COMPOSANTES PRINCIPALES
# Projeter les données sur les composantes principales

# Composantes principales et projections
_, principal_components = np.linalg.eig(X_corrcoef)
data_pca = np.dot(X_standardise, principal_components)
#Correlation
correlation_with_axes = np.corrcoef(X, data_pca, rowvar=False)[:X.shape[1], X.shape[1]:]
#  Tableau
print("Corrélation entre chaque variable et les composantes principales (4 premiers axes) :\n")
print("Variable\tAxe 1\tAxe 2\tAxe 3\tAxe 4")
for i in range(X.shape[1]):
    print(f"Variable {i+1}\t", end="")
    print("\t".join(f"{correlation_with_axes[i, j]:.4f}" for j in range(4)))

####################################################################################################
                            # Analyse des contribution à la variance de CP

trace_correlation = np.trace(Correlation)
first_four_eigenvalues = eigenvalues[:4]
variance_explained_ratios_1 = first_four_eigenvalues[0] / trace_correlation
variance_explained_ratios_2 = first_four_eigenvalues[1] / trace_correlation
variance_explained_ratios_3 = first_four_eigenvalues[2] / trace_correlation
variance_explained_ratios_4 = first_four_eigenvalues[3] / trace_correlation
# Cotisation à la variance
pourcentages_variance = (valeurs_propres / np.sum(valeurs_propres)) * 100
# Calcul de la variance cumulée
variance_cumulee = np.cumsum(pourcentages_variance)
#   tableau récapitulatif
tableau_recapitulatif = np.column_stack((valeurs_propres, pourcentages_variance, variance_cumulee))
tableau_recapitulatif = np.insert(tableau_recapitulatif, 0, range(1, len(eigenvalues) + 1), axis=1)
print("Composante | Valeur Propre | Pourcentage de Variance | Variance Cumulée")
print("-" * 65)
for row in tableau_recapitulatif:
    print(f"{int(row[0]):^10d} | {row[1]:^13.4f} | {row[2]:^23.4f} | {row[3]:^16.4f}")
print("4 premières valeurs propres :", first_four_eigenvalues)
# print("Taux de variance expliquée pour les 4 premières composantes principales :", variance_explained_ratios)
print("somme des variances exprimées")
sum_variance_explained = np.sum(variance_explained_ratios_1 +variance_explained_ratios_2 + variance_explained_ratios_3 +variance_explained_ratios_4)
print(sum_variance_explained)

##########################################################################################################################################

                                # CHARGES FACTORIELLES : Vérification en utilisant directement la librairie PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler

# Standardiser les données
scaler = StandardScaler()
X_standardise = scaler.fit_transform(X)
# Effectuer l'ACP sur les données standardisées
pca = PCA()
pca.fit(X_standardise)
print(pca.explained_variance_ratio_)
nombre_composantes = 4
pca = PCA(n_components=nombre_composantes)
pca.fit(X_standardise)
# Obtenir les coordonnées factorielles pour toutes les observations sur les axes principaux
coordonnees_factorielles = pca.transform(X_standardise)
#  un tableau de corrélation
correlation_table = pd.DataFrame(index=[f'Variable {j + 1}' for j in range(X.shape[1])],
                                 columns=[f'Axe principal {i + 1}' for i in range(nombre_composantes)])
for i in range(nombre_composantes):
    for j in range(X.shape[1]):
        correlation_table.iloc[j, i] = np.corrcoef(coordonnees_factorielles[:, i], X_standardise[:, j])[0, 1]
print("Individual PCA")
print(correlation_table)
print("visualisation graphique")

# graphique de corrélation
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(nombre_composantes):
    ax.bar(range(X.shape[1]),
           [np.corrcoef(coordonnees_factorielles[:, i], X_standardise[:, j])[0, 1] for j in range(X.shape[1])],
           label=f'Axe principal {i + 1}', alpha=0.7)
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels([f'Variable {j + 1}' for j in range(X.shape[1])])
ax.set_ylabel('Corrélation')
ax.legend()
plt.title('Corrélation entre les Axes Principaux et les Variables')
plt.show()
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
##############################################################################################################

                            # Nos vérification étant fructueuses, on peut se lancer dans l'analyse graphique


# PC1 vs PC2
eigenvalues, eigenvectors = np.linalg.eig(X_corrcoef)
# indices du tableau trié des valeurs propres
sorted_indices = np.argsort(eigenvalues)[::-1]
# Trier les valeurs propres et les vecteurs propres
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
Y = eigenvectors
#graphique
axs[0, 0].scatter(coordonnees_factorielles[:, 0], coordonnees_factorielles[:, 1])
for i, txt in enumerate(range(1, len(coordonnees_factorielles) + 1)):
    axs[0, 0].annotate(f'i{txt}', (coordonnees_factorielles[i, 0], coordonnees_factorielles[i, 1]))
arrow_scale = 2
for i in range(Y.shape[0]):
    axs[0, 0].arrow(0, 0, Y[i, 0] * arrow_scale, Y[i, 1] * arrow_scale, color='r', width=0.005, head_width=0.2,
                    head_length=0.2)
    axs[0, 0].text(Y[i, 0] * arrow_scale, Y[i, 1] * arrow_scale, noms_variables[i], ha='right', va='bottom', fontsize=8)
axs[0, 0].set_title('PC1 vs PC2')
axs[0, 0].set_xlabel('PC1')
axs[0, 0].set_ylabel('PC2')

# PC1 vs PC3
axs[0, 1].scatter(coordonnees_factorielles[:, 0], coordonnees_factorielles[:, 2])
for i, txt in enumerate(range(1, len(coordonnees_factorielles) + 1)):
    axs[0, 1].annotate(f'i{txt}', (coordonnees_factorielles[i, 0], coordonnees_factorielles[i, 2]))
# Lignes représentant les variables
for i in range(Y.shape[0]):
    axs[0, 1].arrow(0, 0, Y[i, 0] * arrow_scale, Y[i, 2] * arrow_scale, color='r', width=0.005, head_width=0.2,
                    head_length=0.2)
    axs[0, 1].text(Y[i, 0] * arrow_scale, Y[i, 2] * arrow_scale, noms_variables[i], ha='right', va='bottom', fontsize=8)
axs[0, 1].set_title('PC1 vs PC3')
axs[0, 1].set_xlabel('PC1')
axs[0, 1].set_ylabel('PC3')

# PC2 vs PC3
axs[1, 0].scatter(coordonnees_factorielles[:, 1], coordonnees_factorielles[:, 2])
for i, txt in enumerate(range(1, len(coordonnees_factorielles) + 1)):
    axs[1, 0].annotate(f'i{txt}', (coordonnees_factorielles[i, 1], coordonnees_factorielles[i, 2]))
for i in range(Y.shape[0]):
    axs[1, 0].arrow(0, 0, Y[i, 1] * arrow_scale, Y[i, 2] * arrow_scale, color='r', width=0.005, head_width=0.2,
                    head_length=0.2)
    axs[1, 0].text(Y[i, 1] * arrow_scale, Y[i, 2] * arrow_scale, noms_variables[i], ha='right', va='bottom', fontsize=8)
axs[1, 0].set_title('PC2 vs PC3')
axs[1, 0].set_xlabel('PC2')
axs[1, 0].set_ylabel('PC3')
plt.tight_layout()
plt.show()
correlation_matrix = np.corrcoef(X, rowvar=False)
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
cbar = plt.colorbar()
cbar.set_label('Coefficients de Corrélation', rotation=270, labelpad=15)
for i in range(len(noms_variables)):
    for j in range(len(noms_variables)):
        plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color='black')
plt.xticks(range(len(noms_variables)), noms_variables)
plt.yticks(range(len(noms_variables)), noms_variables)

#Matrice de corrélation coloriée (graphique)
plt.title('Matrice des Coefficients de Corrélation')
plt.show()
#################################################################################################################
                                            #QUESTION 1.11

#ACP pour la Question 1.11


#: Soit un nuage de points de 3 individus

#Nos variables :
D_ronde = np.array([[0, 1, 2],
                    [1, 0, 1],
                    [2, 1, 0]])

D = np.array([[1/3, 0, 0],
              [0, 1/3, 0],
              [0, 0,1/3]])

unite_matrix_scaled =  np.array([[1/3, 1/3, 1/3],
              [1/3, 1/3, 1/3],
              [1/3, 1/3, 1/3]])

W = -1/2 * (D_ronde - D_ronde@unite_matrix_scaled -unite_matrix_scaled@D_ronde +(unite_matrix_scaled@D_ronde@ unite_matrix_scaled))
print("Matrice W:")
print(W)
WD = np.dot(W,D)

print("Matrice WD:")
print(WD, "XD")
valeurs_propres, vecteurs_propres = np.linalg.eig(WD)

# Afficher les résultats
print("Valeurs propres:")
print(valeurs_propres)

print("\nVecteurs propres:")
print(vecteurs_propres)