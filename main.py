"""-------------imports-------------"""
import pandas
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import sklearn
from sklearn import metrics
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(123)
pd.set_option('display.max_columns', None)

"""---------------------Upload and load df---------------------"""

# Load data
data = pd.read_csv("data.csv", header=0)
# Separate data to categorical and continuous variables
data_categorical = data[['country', 'continent']]
data_continuous = data.drop(['continent'], 1)
data_y = data["happiness"]
nrow = len(data)
#print("There are ", nrow, " countries in the database", sep='')
# as dataframe
df = pd.DataFrame(data,
                  columns=['country', 'happiness', 'GDP', 'GDP_per_capita', 'population', 'density', 'unemployment',
                           'life_expectancy', 'out_of_school', 'military_exp', 'health_exp', 'HIV', 'urban', 'suicide',
                           'continent', 'temp', 'fertility'])

#print(df)

"""---------------------Data Understanding---------------------"""

print("Description of the categorial data:")
print(data_categorical.describe())
print("Description of the continuous data:")
print(data_continuous.describe())


"""------Prior probability------"""


### Prior probability - Histogram of Continent ###
print("\nDescription of continent:")
print(data['continent'].describe())
# plt.figure(figsize=(5,5))
ax = sns.countplot(x="continent", data=data)
labels = ("Asia", "Europe", "Africa", "North\nAmerica", "South\nAmerica", "Oceania")
ax.set_xticklabels(labels)
plt.title('Histogram - Continent')
plt.xlabel(' ')
plt.ylabel('Count')
# plt.legend()
plt.ylim(0,70)
# textstr = '\n'.join((
#  r'As = Asia',
#  r'Eu = Europe',
#  r'Af = Africa',
#  r'NA = North America',
#  r'SA = South America',
#  r'Aus = Australia',
#  r'CA = Central America',
#  r'Oc = Oceania'))  # for text box (the content)
for p in ax.patches:
 x = p.get_bbox().get_points()[:,0]
 y = p.get_bbox().get_points()[1,1]
 ax.annotate('{:.1f}%'.format(100.*y/nrow), (x.mean(), y),
             ha='center', va='bottom')  # set the alignment of the text and the %
# ax.yaxis.set_major_locator(ticker.LinearLocator(11))  # the scale of the y axis
# ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#      verticalalignment='top', horizontalalignment='left')  # for text box
plt.show()
# plt.savefig("/Histogram_continent.png")



### Prior probability - Happiness ###
print("\nDescription of Happiness:")
print(data['happiness'].describe())
# Density
#plt.xlim(0,10)
plt.xlabel('Happiness')
plt.ylabel('Density')
plt.title("Happiness - Density Plot")
sns.kdeplot(data['happiness'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,10)
sns.boxplot(x=data['happiness'])
plt.title("Happiness - Box Plot")
plt.show()

### Prior probability - GDP ###
print("\nDescription of GDP:")
print(data['GDP'].describe())
# Density
plt.xlim(0,8000000000000)
plt.xlabel('GDP')
plt.ylabel('Density')
plt.title("GDP - Density Plot")
sns.kdeplot(data['GDP'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['GDP'])
plt.title("GDP - Box Plot")
plt.show()

### Prior probability - GDP per capita ###
print("\nDescription of GDP per Capita:")
print(data['GDP_per_capita'].describe())
# Density
plt.xlim(0,140000)
plt.xlabel('GDP per capita')
plt.ylabel('Density')
plt.title("GDP per capita - Density Plot")
sns.kdeplot(data['GDP_per_capita'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['GDP_per_capita'])
plt.title("GDP per capita - Box Plot")
plt.show()

### Prior probability - Total Population ###
print("\nDescription of Total Population:")
print(data['population'].describe())
# Density
plt.xlim(0,1500000000)
plt.xlabel('Total Population')
plt.ylabel('Density')
plt.title("Total Population - Density Plot")
sns.kdeplot(data['population'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['population'])
plt.title("Total Population - Box Plot")
plt.show()

### Prior probability - Density of Population ###
print("\nDescription of density of Population:")
print(data['density'].describe())
# Density
plt.xlim(0,3000)
plt.xlabel('Density of population')
plt.ylabel('Density')
plt.title("Density of Population - Density Plot")
sns.kdeplot(data['density'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['density'])
plt.title("Density of Population - Box Plot")
plt.show()

### Prior probability - Unemployment ###
print("\nDescription of unemployment:")
print(data['unemployment'].describe())
# Density
plt.xlim(0,40)
plt.xlabel('Unemployment')
plt.ylabel('Density')
plt.title("Unemployment - Density Plot")
sns.kdeplot(data['unemployment'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['unemployment'])
plt.title("Unemployment - Box Plot")
plt.show()

### Prior probability - Life Expectancy ###
print("\nDescription of life Expectancy:")
print(data['life_expectancy'].describe())
# Density
#plt.xlim(0,140000000)
plt.xlabel('Life Expectancy')
plt.ylabel('Density')
plt.title("Life Expectancy - Density Plot")
sns.kdeplot(data['life_expectancy'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['life_expectancy'])
plt.title("Life Expectancy - Box Plot")
plt.show()

### Prior probability - Children Out of School ###
print("\nDescription of Children Out of School:")
print(data['out_of_school'].describe())
# Density
plt.xlim(0,8000000)
plt.xlabel('Children Out of School')
plt.ylabel('Density')
plt.title("Children Out of School - Density Plot")
sns.kdeplot(data['out_of_school'],shade=True)
plt.show()
# Box Plot
plt.xlim(0,8000000)
sns.boxplot(x=data['out_of_school'])
plt.title("Children Out of School - Box Plot")
plt.show()

### Prior probability - Military Expenditure ###
print("\nDescription of Military Expenditure:")
print(data['military_exp'].describe())
# Density
plt.xlim(0,40)
plt.xlabel('Military Expenditure')
plt.ylabel('Density')
plt.title("Military Expenditure - Density Plot")
sns.kdeplot(data['military_exp'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['military_exp'])
plt.title("Military Expenditure - Box Plot")
plt.show()

### Prior probability - Health Expenditure ###
print("\nDescription of Health Expenditure:")
print(data['health_exp'].describe())
# Density
plt.xlim(0,13000)
plt.xlabel('Health Expenditure')
plt.ylabel('Density')
plt.title("Health Expenditure - Density Plot")
sns.kdeplot(data['health_exp'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['health_exp'])
plt.title("Health Expenditure - Box Plot")
plt.show()

### Prior probability - Prevalence of HIV ###
print("\nDescription of Prevalence of HIV:")
print(data['HIV'].describe())
# Density
plt.xlim(0,35)
plt.xlabel('Prevalence of HIV')
plt.ylabel('Density')
plt.title("Prevalence of HIV - Density Plot")
sns.kdeplot(data['HIV'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['HIV'])
plt.title("Prevalence of HIV - Box Plot")
plt.show()

### Prior probability - Urban Population ###
print("\nDescription of Urban Population:")
print(data['urban'].describe())
# Density
plt.xlim(0,130)
plt.xlabel('Urban Population')
plt.ylabel('Density')
plt.title("Urban Population - Density Plot")
sns.kdeplot(data['urban'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['urban'])
plt.title("Urban Population - Box Plot")
plt.show()


### Prior probability - Suicide Mortality Rate ###
print("\nDescription of Suicide Mortality Rate:")
print(data['suicide'].describe())
# Density
plt.xlim(0,80)
plt.xlabel('Suicide Mortality Rate')
plt.ylabel('Density')
plt.title("Suicide Mortality Rate - Density Plot")
sns.kdeplot(data['suicide'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['suicide'])
plt.title("Suicide Mortality Rate - Box Plot")
plt.show()


### Prior probability - Temperature ###
print("\nDescription of Temperature:")
print(data['temp'].describe())
# Density
#plt.xlim(0,140000000)
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.title("Temperature - Density Plot")
sns.kdeplot(data['temp'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['temp'])
plt.title("Temperature - Box Plot")
plt.show()

### Prior probability - Fertility Rate ###
print("\nDescription of Fertility Rate:")
print(data['fertility'].describe())
# Density
plt.xlim(0,8)
plt.xlabel('Fertility Rate')
plt.ylabel('Density')
plt.title("Fertility Rate - Density Plot")
sns.kdeplot(data['fertility'],shade=True)
plt.show()
# Box Plot
#plt.xlim(0,140000000)
sns.boxplot(x=data['fertility'])
plt.title("Fertility Rate - Box Plot")
plt.show()
"""
"""
### scatter plot - Health Expenditure vs GDP per Capita
plt.scatter(df.health_exp, df.GDP_per_capita)
plt.xlabel('Health Expenditure')
plt.ylabel('GDP per Capita')
plt.title("Scatter Plot - Health Expenditure vs GDP per Capita")
plt.tight_layout()  # so the labels won't be cut-off
plt.show()


"""------Missing Values------"""


print("Number of missing values in the table:",data.isnull().sum().sum())

# missing value - each col:
columns = list(data)
columns.pop(0)  # without "country" column
columns.pop(0)  # without "happiness" column

for col in columns:
    missing = data[col].isnull().sum().sum()
    missing_precentage = round(missing * 100 / nrow, 1)
    print("Missing values in column '",col,"': ",missing,", which is ",missing_precentage,"%",sep = '')


"""------Correlations------"""


# Heatmap - between all continuous variables
correlation = df.drop('country',1)
correlation = correlation.corr()
axis_labels = ['happiness score', 'GDP', 'GDP per capita', 'Total Population', 'Density of Population', 'Unemployment', 'Life expectancy', 'Children out of school',
               'Military expenditure', 'Health Expenditure', 'Prevalence of HIV', 'Urban Population', 'Suicide mortality rate', 'Temperature', 'Fertility Rate']
sns.set(font_scale=0.9)  # font size
plt.figure(figsize=(12, 9))
sns.heatmap(correlation, annot=True, linewidths=1, cmap='coolwarm', square=True, xticklabels=axis_labels, yticklabels=axis_labels)
plt.tight_layout()  # so the labels won't be cut-off
plt.show()

# Pairplot - between all continuous variables
sns.set_context("paper", rc={"axes.labelsize":20})
sns.pairplot(df, height=2)
plt.show()

# Categorial
# continent with Happiness: (categorical vs continuous) - scatter
catplot_cont = sns.catplot(x=df['continent'], y=df['happiness'], data=df)
catplot_cont.set(title="Continent vs Happiness", xlabel=None, ylabel=None)
labels = ("Asia", "Europe", "South\nAmerica", "Oceania", "Africa", "North\nAmerica")
catplot_cont.set_xticklabels(labels)
plt.tight_layout()
plt.ylabel('Happiness')
plt.xlabel('Continent')
plt.tight_layout()  # so the labels won't be cut-off
plt.show()

# continent with Happiness: (categorical vs continuous) - box plot
sns.catplot(data=df, x = "continent", y = "happiness", kind="box" , aspect=1.2)
plt.show()

"""---------------------Data Preperation---------------------"""

new_df = df.copy()
new_df = new_df.drop(columns=['GDP_per_capita', 'density', 'military_exp', 'population'])
#new_df = new_df.drop(columns=['suicide','unemployment'])

### Missing Values

print("Number of missing values in the table:",new_df.isnull().sum().sum())

# missing value - each col:
columns = list(new_df)
columns.pop(0)  # without "country" column
columns.pop(0)  # without "happiness" column

for col in columns:
    missing = new_df[col].isnull().sum().sum()
    missing_precentage = round(missing * 100 / nrow, 1)
    print("Missing values in column '",col,"': ",missing,", which is ",missing_precentage,"%",sep = '')

## Remove Rows

new_df = new_df[new_df.country != 'Hong Kong SAR, China']
new_df = new_df[new_df.country != 'Kosovo']
new_df = new_df[new_df.country != 'West Bank and Gaza']

new_df = new_df.reset_index()
new_df = new_df.drop(columns=['index'])

## Remove Columns

new_df = new_df.drop(columns=['out_of_school'])

## Missing Values Completion - from Internet

#GDP
new_df['GDP'] = np.where((new_df['country']=='South Sudan'), 1000000000, new_df['GDP'])
new_df['GDP'] = np.where((new_df['country']=='Venezuela, RB'), 106000000000.36, new_df['GDP'])
new_df['GDP'] = np.where((new_df['country']=='Turkmenistan'), 45000000000.23, new_df['GDP'])

#Health Expenditure
new_df['health_exp'] = np.where((new_df['country']=='Libya'), 171, new_df['health_exp'])
new_df['health_exp'] = np.where((new_df['country']=='Albania'), 275, new_df['health_exp'])
new_df['health_exp'] = np.where((new_df['country']=='Yemen, Rep.'), 57, new_df['health_exp'])

#Prevalence of HIV
new_df['HIV'] = np.where((new_df['country']=='United States'), 0.36, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Sweden'), 0.2, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Canada'), 0.17, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='United Kingdom'), 0.2, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Malta'), 0.088, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Saudi Arabia'), 0.024, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Panama'), 0.8, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Russian Federation'), 1.2, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Bosnia and Herzegovina'), 0.009, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='China'), 0.09, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Armenia'), 0.2, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Mozambique'), 12.1, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Austria'), 0.1, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Belgium'), 0.3, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Finland'), 0.2, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Japan'), 0.1, new_df['HIV'])
new_df['HIV'] = np.where((new_df['country']=='Israel'), 0.2, new_df['HIV'])

#Temperature
new_df['temp'] = np.where((new_df['country']=='South Sudan'), 35, new_df['temp'])

#Fertility Rate
new_df['fertility'] = np.where((new_df['country']=='Korea, Rep.'), 1, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Russian Federation'), 1.6, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Turkiye'), 2.1, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Egypt, Arab Rep.'), 3.3, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Czechia'), 1.7, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Iran, Islamic Rep.'), 2.1, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Slovak Republic'), 1.5, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Cote d\'Ivoire'), 4.6, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Congo, Dem. Rep.'), 5.9, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Lao PDR'), 2.7, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Yemen, Rep.'), 3.8, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Congo, Rep.'), 4.4, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Kyrgyz Republic'), 3.3, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Gambia, The'), 5.2, new_df['fertility'])
new_df['fertility'] = np.where((new_df['country']=='Venezuela, RB'), 2.3, new_df['fertility'])


print("Number of missing values in the table:",new_df.isnull().sum().sum())

# missing value - each col:
columns = list(new_df)
columns.pop(0)  # without "country" column
columns.pop(0)  # without "happiness" column

for col in columns:
    missing = new_df[col].isnull().sum().sum()
    missing_precentage = round(missing * 100 / nrow, 1)
    print("Missing values in column '",col,"': ",missing,", which is ",missing_precentage,"%",sep = '')


## Missing Values Completion - other ways

#HIV
new_df['HIV'] = np.where((new_df['HIV'].isnull()), (-2.818+0.313*new_df['unemployment']+0.22*new_df['suicide']), new_df['HIV'])
new_df = new_df.drop(columns=['suicide','unemployment'])
new_df['HIV'][new_df['HIV'] < 0] = 0

### Categories Unification

#continent

c = ['tab:red', 'tab:green', 'tab:purple', 'tab:blue', 'tab:orange','tab:pink']
ax = new_df['continent'].value_counts().plot(kind='bar', figsize=(8, 6), width = 0.8, color = c, rot=0)
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f')
plt.xlabel("continent",fontweight='bold')
plt.ylabel("# Countries",fontweight='bold')
plt.title("Bar Graph of Continent - Before Unification", y = 1.02, fontweight='bold')
plt.tight_layout()
plt.show()

new_df['continent'] = new_df['continent'].replace(['South America'], 'America')
new_df['continent'] = new_df['continent'].replace(['North America'], 'America')
new_df['continent'] = new_df['continent'].replace(['Oceania'], 'Asia')

c = ['tab:red', 'tab:green', 'tab:purple', 'tab:blue', 'tab:orange','tab:pink']
ax = new_df['continent'].value_counts().plot(kind='bar', figsize=(8, 6), width = 0.8, color = c, rot=0)
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f')
plt.xlabel("continent",fontweight='bold')
plt.ylabel("# Countries",fontweight='bold')
plt.title("Bar Graph of Continent - After Unification", y = 1.02, fontweight='bold')
plt.tight_layout()
plt.show()

### Normalizing

country_col = new_df.country
continent_col = new_df.continent
norm_df = new_df.drop(["country","continent"], axis=1)

# robustScaler
transformer = RobustScaler().fit(norm_df)
RobustScaler()
norm_df = transformer.transform(norm_df)
norm_df = pd.DataFrame(data=norm_df,columns=['happiness', 'GDP', 'life_expectancy', 'health_exp', 'HIV', 'urban', 'temp', 'fertility'])

# min max
scaler = MinMaxScaler()
norm_df = scaler.fit_transform(norm_df)
norm_df = pd.DataFrame(data=norm_df,columns=['happiness', 'GDP', 'life_expectancy', 'health_exp', 'HIV', 'urban', 'temp', 'fertility'])

norm_df.insert(len(norm_df.columns), "continent", continent_col, True)  # insert continent col

# create dummies for continent
norm_df_dumm = pd.get_dummies(norm_df, columns = ['continent'])

# Separating x and y variables
norm_df_dummY = norm_df_dumm.happiness
norm_df_dummX = norm_df_dumm.drop(["happiness"],axis=1)

#with continent
#norm_df.insert(0, "country", country_col, True)  # insert country col

"""---------------------Modeling---------------------"""

### Decide K for clustering
iner_list = []
dbi_list = []
sil_list = []
for n_clusters in range(2, 10, 1):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=500, n_init=100, random_state=1)
    kmeans.fit(norm_df_dummX, norm_df_dummY)
    assignment = kmeans.predict(norm_df_dummX)
    iner = kmeans.inertia_
    sil = sklearn.metrics.silhouette_score(norm_df_dummX, assignment)
    dbi = sklearn.metrics.davies_bouldin_score(norm_df_dummX, assignment)
    dbi_list.append(dbi)
    sil_list.append(sil)
    iner_list.append(iner)


# Davies–Bouldin
plt.plot(range(2, 10, 1), dbi_list, marker='o')
plt.title("Davies–Bouldin")
plt.xlabel("Number of clusters")
plt.ylabel("Davies–Bouldin Values")
plt.show()
# Silhouette
plt.plot(range(2, 10, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Values")
plt.show()
# iner - the elbow method
plt.plot(range(2, 10, 1), iner_list, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()


### Hierarchical Clustering

# Dendogram for Hierarchical Clustering
temp = norm_df_dumm.drop(["happiness"],axis=1)
plt.figure(figsize=(20, 15))
plt.title("Dendogram")
dend = shc.dendrogram(shc.linkage(temp, method='ward'))
plt.show()


Hierarchical = AgglomerativeClustering(n_clusters=4)
norm_df = norm_df.drop(["continent"],axis=1)
Hierarchical.fit(norm_df)
norm_df['Hierarch_clus'] = Hierarchical.fit_predict(norm_df)

##with continent
#Hierarchical.fit(norm_df_dumm)
#norm_df_dumm['Hierarch_clus'] = Hierarchical.fit_predict(norm_df_dumm)
#norm_df['Hierarch_clus'] = Hierarchical.fit_predict(norm_df_dumm)

### PCA

##with continent
#df_pca = norm_df.drop(["happiness","Hierarch_clus"],axis=1)

df_pca = norm_df.drop(["happiness", "Hierarch_clus"],axis=1)
#PCA on x (no happiness) without categorial variables
pca = PCA(n_components=2)
#fit and not for transform for explained_variance_ratio_
pca = pca.fit(df_pca)
print("explained variance with pca:",pca.explained_variance_ratio_)
#fit transform
pca = pca.fit_transform(df_pca)
df_pca = pd.DataFrame(pca, columns=['PCA1', 'PCA2'])

# df_pca = pca1, pca2 and hierarch result
df_pca.insert(len(df_pca.columns), column='Hierarch_clus', value=norm_df.Hierarch_clus)
new_df.insert(len(new_df.columns), column='Hierarch_clus', value=norm_df.Hierarch_clus)


sns.lmplot(data=df_pca,x='PCA1',y='PCA2', hue='Hierarch_clus', fit_reg=False, legend=False)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title("Clusters of Hierarchical Clustering (4)")
plt.legend(loc='best')
plt.legend(prop={'size': 6})
plt.tight_layout()
plt.show()

df_pca.insert(0, column='country', value=country_col)
df_pca.insert(1, column='happiness', value=norm_df.happiness)


# (Hierarchical) histogram of each cluster of happiness, all together - COUNT
for i in range(4): # 4 clustering
    which_cluster = df_pca[df_pca["Hierarch_clus"] == i]
    if i == 0:
        s = '0'
        color = 'blue'
    if i == 1:
        s = '1'
        color = 'orange'
    if i == 2:
        s = '2'
        color = 'green'
    if i == 3:
        s = '3'
        color = 'red'
    sns.distplot(which_cluster['happiness'], hist=True, kde=False, norm_hist=False,
                 bins=int(180 / 10), color=color, hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4}, label=s)
plt.title("Histogram of Happiness by Hierarchical Clustering (Count)")
plt.xlabel('Happiness')
plt.ylabel('Count')
plt.legend(loc='best')
plt.legend(title="Clusters", labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.tight_layout()
plt.show()

# (Hierarchical) histogram of each cluster of gdp, all together - DENSITY
for i in range(4): # 4 clustering
    which_cluster = df_pca[df_pca["Hierarch_clus"] == i]
    if i == 0:
        s = '0'
        color = 'blue'
    if i == 1:
        s = '1'
        color = 'orange'
    if i == 2:
        s = '2'
        color = 'green'
    if i == 3:
        s = '3'
        color = 'red'
    sns.distplot(which_cluster['happiness'], hist=False, kde=True,
                 bins=int(180 / 10), color=color, hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4}, label=s)
plt.title("Histogram of Happiness by Hierarchical Clustering (Density)")
plt.xlabel('Happiness')
plt.ylabel('Density')
plt.legend(loc='best')
plt.legend(title="Clusters", labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
plt.tight_layout()
plt.show()

# (Hierarchical) descriptive statistics for Happiness for each cluster
for i in range(4): # all 4 clusters
    which_cluster = new_df[new_df["Hierarch_clus"] == i]
    print("\nDescriptive data for cluster number ",i,":",sep='')

    print("happiness: mean",which_cluster['happiness'].mean())
    #print("happiness: std",which_cluster['happiness'].std())

    print("GDP: mean", which_cluster['GDP'].mean())
    #print("GDP: std", which_cluster['GDP'].std())

    print("life_expectancy: mean", which_cluster['life_expectancy'].mean())
    #print("life_expectancy: std", which_cluster['life_expectancy'].std())

    print("health_exp: mean", which_cluster['health_exp'].mean())
    #print("health_exp: std", which_cluster['health_exp'].std())

    print("HIV: mean", which_cluster['HIV'].mean())
    #print("HIV: std", which_cluster['HIV'].std())

    print("urban: mean", which_cluster['urban'].mean())
    #print("urban: std", which_cluster['urban'].std())

    print("temp: mean", which_cluster['temp'].mean())
    #print("temp: std", which_cluster['temp'].std())

    print("fertility: mean", which_cluster['fertility'].mean())
    #print("fertility: std", which_cluster['fertility'].std())


for i in range(4): # all 4 clusters
    which_cluster = new_df[new_df["Hierarch_clus"] == i]
    print("\nContinent in cluster number ",i,":",sep='')
    print(which_cluster['continent'].describe())


#drop Hierarch_clus before kmeams
Hierarch_clus = norm_df.Hierarch_clus
norm_df = norm_df.drop(["Hierarch_clus"],axis=1)

### K-Means

kmeans = KMeans(n_clusters=4, random_state=421)
k_model = kmeans.fit(norm_df)
norm_df['KMeans_clus'] = k_model.predict(norm_df)

#return Hierarch_clus inside
norm_df.insert(len(norm_df.columns), column='Hierarch_clus', value=Hierarch_clus)

df_pca.insert(len(df_pca.columns), column='KMeans_clus', value=norm_df.KMeans_clus)
new_df.insert(len(new_df.columns), column='KMeans_clus', value=norm_df.KMeans_clus)


sns.lmplot(data=df_pca, x='PCA1', y='PCA2', hue='KMeans_clus', fit_reg=False, legend=False)
plt.xlabel('PCA1', fontsize=12)  # PCA1
plt.ylabel('PCA2', fontsize=12)  # PCA2
plt.title("Clusters of K-Means Clustering (4)")
plt.legend(loc='best')
plt.legend(title="Clusters", labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.legend(loc=2, prop={'size': 6})
plt.tight_layout()
plt.show()

# (kmeans) histogram of each cluster of happiness- DENSITY
for i in range(4): # 4 clustering
    which_cluster = df_pca[df_pca["KMeans_clus"] == i]
    if i == 0:
        s = '0'
        color = 'blue'
    if i == 1:
        s = '1'
        color = 'orange'
    if i == 2:
        s = '2'
        color = 'green'
    if i == 3:
        s = '3'
        color = 'red'
    sns.distplot(which_cluster['happiness'], hist=False, kde=True,
                 bins=int(180 / 10), color=color, hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4}, label=s)
plt.title("Histogram of Happiness by K-Means Clustering (Count)")
plt.xlabel('Happiness', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(loc='best')
plt.legend(title="Clusters", labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.tight_layout()
plt.show()

# (kmeans) descriptive statistics for Happiness for each cluster
arr2 = []
# not normalize, KMeans_clus
for i in range(4): # all 4 clusters
    temp_arr = []
    which_cluster = new_df[new_df["KMeans_clus"] == i]
    temp_arr.insert(len(temp_arr),which_cluster['happiness'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['GDP'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['life_expectancy'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['health_exp'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['HIV'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['urban'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['temp'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['fertility'].mean())
    arr2.insert(len(arr2),temp_arr)
print(arr2)


for i in range(4): # all 4 clusters
    which_cluster = new_df[new_df["KMeans_clus"] == i]
    print("\nContinent in cluster number ",i,":",sep='')
    print(which_cluster['continent'].describe())

### metrices for clustering

labels = k_model.labels_
silhouette_score = metrics.silhouette_score(norm_df,labels)
print("silhouette score - kmeans clustering: ",silhouette_score)

labels = Hierarchical.labels_
silhouette_score = metrics.silhouette_score(norm_df,labels)
print("silhouette score - Hierarchical clustering: ",silhouette_score)

labels = k_model.labels_
calinski_harabasz_score = metrics.calinski_harabasz_score(norm_df, labels)
print("calinski harabasz score - kmeans clustering: ",calinski_harabasz_score)

labels = Hierarchical.labels_
calinski_harabasz_score = metrics.calinski_harabasz_score(norm_df, labels)
print("calinski harabasz score - Hierarchical clustering: ",calinski_harabasz_score)

labels = k_model.labels_
davies_bouldin_score = metrics.davies_bouldin_score(norm_df, labels)
print("davies bouldin score - kmeans clustering: ",davies_bouldin_score)

labels = Hierarchical.labels_
davies_bouldin_score = metrics.davies_bouldin_score(norm_df, labels)
print("davies bouldin score - Hierarchical clustering: ",davies_bouldin_score)


### Radar plot

arr1 = []
# NORMALIZED for Hierarch_clus
for i in range(4): # all 4 clusters
    temp_arr = []
    which_cluster = norm_df[norm_df["Hierarch_clus"] == i]
    temp_arr.insert(len(temp_arr),which_cluster['happiness'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['GDP'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['life_expectancy'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['health_exp'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['HIV'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['urban'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['temp'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['fertility'].mean())
    arr1.insert(len(arr1),temp_arr)
print(arr1)

arr2 = []
# NORMALIZED for KMeans_clus
for i in range(4): # all 4 clusters
    temp_arr = []
    which_cluster = norm_df[norm_df["KMeans_clus"] == i]
    temp_arr.insert(len(temp_arr),which_cluster['happiness'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['GDP'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['life_expectancy'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['health_exp'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['HIV'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['urban'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['temp'].mean())
    temp_arr.insert(len(temp_arr), which_cluster['fertility'].mean())
    arr2.insert(len(arr2),temp_arr)
print(arr2)


#Radar plot - Hierarchy
categories = ['happiness','GDP','life expectancy','health expenditure','HIV', 'urban', 'temp','fertility']
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r=arr1[0],
      theta=categories,
      fill='toself',
      name='0'
))
fig.add_trace(go.Scatterpolar(
      r=arr1[1],
      theta=categories,
      fill='toself',
      name='1'
))
fig.add_trace(go.Scatterpolar(
      r=arr1[2],
      theta=categories,
      fill='toself',
      name='2'
))
fig.add_trace(go.Scatterpolar(
      r=arr1[3],
      theta=categories,
      fill='toself',
      name='3'
))
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=False,
      range=[0, 1]
    )),
  showlegend=True
)
fig.show()


#Radar plot - KMeans
categories = ['happiness','GDP','life expectancy','health expenditure','HIV', 'urban', 'temp','fertility']
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r=arr2[0],
      theta=categories,
      fill='toself',
      name='0'
))
fig.add_trace(go.Scatterpolar(
      r=arr2[1],
      theta=categories,
      fill='toself',
      name='1'
))
fig.add_trace(go.Scatterpolar(
      r=arr2[2],
      theta=categories,
      fill='toself',
      name='2'
))
fig.add_trace(go.Scatterpolar(
      r=arr2[3],
      theta=categories,
      fill='toself',
      name='3'
))
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=False,
      range=[0, 1]
    )),
  showlegend=True
)
fig.show()

#continent
continent_data = pd.DataFrame(data=new_df,columns=['country', 'continent', 'KMeans_clus'])

pandas.set_option('display.max_rows', 151)
print(continent_data)