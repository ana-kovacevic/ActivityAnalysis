'''
########################################################################################
MAPPING CLUSTERS TO GRADES - SINGLE VARIATE CASE - 5 CLUSTERS TO 5 GRADES
########################################################################################
'''
#Creates map (dictionary) between clusters (cluster numbers) and their means
def create_map__means_to_clusters(model):
    means = [model.means_[i][0] for i in range(len(model.means_))]
    clusters=list(range(len(model.means_)))
    return dict(zip(clusters, means))

# Creates map (dictionary) from cluster means to grades 1-5. Based on extremization type, sorts means and assigns grades
def create_map_means_to_grades(model, activity, activity_extremization):
    extrem=activity_extremization
    means=[model.means_[i][0] for i in range(len(model.means_))]
    sorted = np.sort(means)
    if extrem == 'max':
        grades=range(1, len(sorted)+1)
    else:
        grades = range(len(sorted),0,-1)
    return(dict(zip(grades, sorted)))

# Creates map from clusters to grades - this map is input for mapping all time points (clusters) to grades
def create_map_clusters_to_grades(map_grades, map_clusters):
    map_clusters_grades={}
    for key_clust, value_clust in map_clusters.items():
        for key_grade, value_grade in map_grades.items():
            if value_grade == value_clust:
                map_clusters_grades.update({key_clust:key_grade})
    return(map_clusters_grades)


### Assigns grades to each cluster and returns map
def map_grades_to_clusters(clusters, map_clusters_grades):
    grades=list(map(lambda x: map_clusters_grades[x], clusters))
    return(grades)

### Calculates grade for each time point based on single variate clusters
### Only this method should be called
def calculate_grades(res):
    activities=res.keys()
    for activity in activities:
        model=res[activity]['model']
        activity=res[activity]['name']
        clusters=res[activity]['clusters']
        activity_extremization=res[activity]['extremization']
        map_grades = create_map_means_to_grades(model, activity, activity_extremization)
        map_clusters = create_map__means_to_clusters(model)
        map_clusters_grades = create_map_clusters_to_grades(map_grades, map_clusters)
        grades = map_grades_to_clusters(clusters, map_clusters_grades)
        res[activity].update({'grades':grades})


### Creates higher factor based on result object
def create_higher_factor(res):
    activities = res.keys()
    key_for_len=list(activities)[0]
    size=len(res[key_for_len]['grades'])
    factor=np.zeros(size)
    for activity in activities:
        weight=res[activity]['weight']
        grades=res[activity]['grades']
        weighted_grades=np.multiply(weight,grades)
        factor=factor+weighted_grades
    return (factor)


'''
########################################################################################
MAPPING CLUSTERS TO GRADES - MULTI VARIATE CASE - 5 CLUSTERS TO 5 GRADES
########################################################################################
'''

#Creates map (dictionary) between clusters (cluster numbers) and their means
# Difference with single variate because it is weighting multivariate clusters in order to get factor from subfactor values


def calculate_factor_weights(model, activity_weights):
    weights = list(activity_weights.values())
    cluster_weighted_means = np.dot(model.means_, weights)  # weighted sum of weights and cluster means for each cluster
    return cluster_weighted_means


def create_map__means_to_clusters(model, cluster_weighted_means):
    number_of_clusters=len(model.means_)
    clusters =list(range(number_of_clusters))
    return dict(zip(clusters, cluster_weighted_means))

# Creates map (dictionary) from cluster means to grades 1-5. Based on extremization type, sorts means and assigns grades
def create_map_means_to_grades(weighted_means):
    means=weighted_means
    sorted = np.sort(means)
    grades=range(1, len(sorted)+1)
    return(dict(zip(grades, sorted)))

# Creates map from clusters to grades - this map is input for mapping all time points (clusters) to grades
def create_map_clusters_to_grades(map_grades, map_clusters):
    map_clusters_grades={}
    for key_clust, value_clust in map_clusters.items():
        for key_grade, value_grade in map_grades.items():
            if value_grade == value_clust:
                map_clusters_grades.update({key_clust:key_grade})
    return(map_clusters_grades)

### Assigns grades to each cluster and returns map
def map_grades_to_clusters(clusters, map_clusters_grades):
    grades=list(map(lambda x: map_clusters_grades[x], clusters))
    return(grades)

### Calculates grade for each time point based on single variate clusters
### Only this method should be called
def calculate_grades(model, clusters, activity_weights):
    weighted_means=calculate_factor_weights(model, activity_weights) # creates higher factor based on weights and multivariate clusters
    map_clusters = create_map__means_to_clusters(model, weighted_means)
    map_grades = create_map_means_to_grades(weighted_means)
    map_clusters_grades = create_map_clusters_to_grades(map_grades, map_clusters)
    grades = map_grades_to_clusters(clusters, map_clusters_grades)
    return grades
