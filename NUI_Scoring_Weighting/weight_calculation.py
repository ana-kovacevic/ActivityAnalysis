'''
################################################################## 
WEIGHT CALCULATION - ALTERNATIVE WAYS TO CALCULATE WEIGHTS BETWEEN SUBFACTORS AND FACTORS
#######################################################################
'''
#Calculates weights based on PCA
#All PCAs are computed for single sub-factor
#All sub-factors are normalized based on components
#All sub-factors are weighted by explained variance of each component
def calculate_measure_weights(pivoted_data):
    from sklearn.decomposition import PCA
    X=pivoted_data.iloc[:,2:]
    pca = PCA(n_components=4)
    pca.fit(X)

    components=np.matrix(np.abs(pca.components_)).transpose()
    normalizer=components.sum(axis=0)
    components=np.divide(components, normalizer)
    weights=np.dot(components,pca.explained_variance_ratio_)
    return weights

