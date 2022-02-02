#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.express as px

import statsmodels.formula.api as sm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, SelectKBest, mutual_info_classif
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_curve, auc, roc_auc_score
from lightgbm import LGBMClassifier

from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline 

from collections import Counter


# ## Pre-processamento e análise dos dados
# Nessa etapa vamos verificar a existência de outliers e se a base de dados está balanceada

# In[2]:


# Importando base de dados
df = pd.read_csv('treino.csv')
df.head()


# In[3]:


# Função para a verificação de dados desbalanceados
def verify_unbalanced(df, col):
    print(f"Total = {len(df)} -> 100%")
    print(f"Bom pagador  = {len(df[df[col] == 0])} -> {len(df[df[col] == 0])/len(df) *100}%")
    print(f"Inadimplente = {len(df[df[col] == 1])} -> {len(df[df[col] == 1])/len(df) *100}%")


# In[4]:


# Verificando desbalanceamento
proporção_classes =  verify_unbalanced(df, 'inadimplente')
proporção_classes


# ### Como pode-se observar a base de dados está severamente desbalanceada
# A proporção de casos de bons pagadores e inadimplentes está severamente desbalanceada, porém antes de lidar com o balanceamento dos dados, vamos verificar a existência de outliers e dados faltantes e trata-los corretamente antes de balancear a base de dados. Assim, agora vamos traçar o perfil dessa base de dados utilizando do pandas profiling e verificar possíveis alertas.

# In[5]:


profile = ProfileReport(df, title='Relatório da base de dados', explorative = True, samples=None,
    correlations=None,
    missing_diagrams=None,
    duplicates=None,
    interactions=None,)
profile


# ## Detecção de  outliers
# Existem várias formas de detecção de outliers, uma delas é verificando o `z_score` de cada valor e se esse valor ultrapassar 3 é dito que esse valor é um outlier do conjunto. Outra abordagem é analizando-se pelo `quantil` de 99% e de 1%, verificando se o  valor é maior que o quantil de 99% ou se é menor que o quantil de 1%.

# In[6]:


def z_score_outlier_count(x):
    z = np.abs(stats.zscore(x))
    threshold = 3
    outliers = [i for i in z if i > threshold]
    return len(outliers)

def quantile_outlier_count(x):
    quantile_99 = x.quantile(0.99)
    count_99 = len([i for i in x if i > quantile_99])

    quantile_01 = x.quantile(0.01)
    count_01 = len([i for i in x if i < quantile_01])
    return count_01 + count_99

def outlier_missing_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), z_score_outlier_count(x), quantile_outlier_count(x)],
                  index=['N', 'N_missing', 'N_outlier', 'N_outlier_quantile'])


# In[7]:


df.apply(lambda x: outlier_missing_summary(x)).T


# In[8]:


#Lidando com outliers
def outlier_capping(x):
    x = x.clip(upper=x.quantile(0.99))
    x = x.clip(lower=x.quantile(0.01))
    return x

def missing_val_treat(x):
    x = x.fillna(x.mean())
    return x


# In[9]:


df = df.apply(lambda x: outlier_capping(x))
df = df.apply(lambda x:  missing_val_treat(x))


# Olhando-se o sumário criado é possível notar a presença dos outliers na base de dados, porém é importante notar que a presença de outliers pode ser utilizada para classificar uma amostra da base de dados. Portanto vamos primeiro avaliar a influencia da presença de outliers nos modelos antes de trata-los.

# ## Análise exploratória
# Nessa etapa será explorado as relações entre as variáveis e possivelmente determinar qual variável é a mais importante para determinar o risco de crédito. Assim, vamos analisar a correlação cruzada entre as nossas features e a nossa label ( coluna de `inadimplentes`).

# In[10]:


fig = px.imshow(df.corr(), text_auto=True ,color_continuous_scale='PuBu', width=1000, height=800, aspect="auto")
fig.show()


# De acordo com a análise com o a correlação cruzada praticamente nenhuma variável apresenta relação forte com a coluna `inadimplentes` e é possível notar que existem 3 colunas com correlação  moderada entre si. Os pares de colunas em questão são: `vezes_passou_de_30_59_dias` e `numero_de_vezes_que_passou_60_89_dias`,  `vezes_passou_de_30_59_dias` e `numero_de_vezes_que_passou_90_dias`, e  `numero_de_vezes_que_passou_60_89_dias` e `numero_de_vezes_que_passou_90_dias`. Assim, esses pares podem apresentar `colinearidade`, indicando que essas colunas passam as mesmas informações, e dessa maneira podem influenciar negativamente os modelos. Portanto antes de retirar alguma dessas colunas da base de teste é importante testar os modelos com e sem as colunas para testar se irá afetar positivamente os modelos de risco de crédito.

# ## Agora vamos analisar a influencia das variáveis na coluna de `inadimplentes`
# Para isso vamos utilizar o `violin plot`, que é capaz de mostrar a densidade de probabilidade de forma semelhante aos graficos de box plot

# In[11]:



for col in list(df.columns):
    print(col)
    if col != 'inadimplente':
        fig = px.violin(df,x='inadimplente', y=f'{col}',color="inadimplente", box=True, width=600, height=600)
        fig.show()


# Analizando-se os gráficos acima é possível notar que a única variável que parece ter um `box plot` e `violin plot` constantes para as duas classes de pagadores é a coluna `razão_debito`, porém como são poucas colunas é vantajoso mantê-la como entrada dos modelos.

# In[12]:


# Separando as entradas e saidas
dataset = df.to_numpy() 
entries = df.loc[:, df.columns != 'inadimplente'].to_numpy(dtype=np.float64)
outputs = df['inadimplente'].to_numpy(dtype=np.int64)
print(entries.shape)
print(outputs.shape)


# In[13]:


# Dividindo o dataset em treino e teste
seed = 10
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(entries, outputs, test_size=test_size, random_state=seed)
print('Train dataset shape:\nEntries: ', x_train.shape, '\nOutput: ', y_train.shape, '\n\n')
print('Test dataset shape:\nEntries: ', x_test.shape, '\nOutput: ', y_test.shape)


# ## Treinando e testando os modelos
# Agora vamos treinar, validar e testar os modelos utilizando o nosso `x_train` e `y_train`. Dessa forma estes serão utilizados para gerar múltiplas `folds` de treino e validação, e posteriormente iremos testa-los com os dados `x_test` e `y_test`.

# In[14]:


def get_classifiers():
    # Decision Tree classifier
    dt = tree.DecisionTreeClassifier(random_state=seed, criterion="entropy", min_samples_leaf=2, min_samples_split=5, max_depth=100)

    # Random forest classifier
    rf = RandomForestClassifier(n_estimators=10, random_state=seed, min_samples_split=5)

    # Naive Bayes classifier
    gnb = GaussianNB()

    #LGBMClassifier
    lgb_class = LGBMClassifier()

    # Elastic net
    elastic_class = LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=1,max_iter=300)

    classifiers = [(dt, "Decision tree"), (gnb, "Naive Bayes"), (rf, "Random forest"), (lgb_class, "LGBMClassifier"), (elastic_class, "Elastic_net")]

    return classifiers

def k_fold_train(classifiers ,x_train, y_train, K=5):
    # Create -fold validation set for training
    kf = StratifiedShuffleSplit(n_splits=K, random_state=seed)

    # Training classifiers using cross-validation
    fold_number = 1
    for train_indexes, valid_indexes in kf.split(x_train, y_train):
        print("Fold ", fold_number)
        for classifier, label in classifiers:
            classifier.fit(x_train[train_indexes], y_train[train_indexes])
            y_valid_pred = classifier.predict(x_train[valid_indexes])
            y_valid_pred = np.round(y_valid_pred)
            print("Classifier type: ",label, ", Validation Accuracy = ", accuracy_score(y_train[valid_indexes], y_valid_pred))
        print('\n')
        fold_number += 1


# In[15]:


classifiers = get_classifiers()
k_fold_train(classifiers ,x_train, y_train)


# ## Testando os modelos
# Para cada modelo instanciado será feita a matrix de confusão assim como serão retiradas métricas de performance.

# In[16]:


def classifiers_test(classifiers, x_test, y_test):
    for classifier, label in classifiers:
        y_test_estimative = classifier.predict(x_test)
        print("Classifier type: ", label, ", Test Accuracy = ", accuracy_score(y_test, y_test_estimative))

classifiers_test(classifiers, x_test, y_test)


# In[17]:


def get_classifiers_cf(classifiers, x_test, y_test):
    total = len(x_test)
    confusion_matrixes = np.zeros((len(classifiers), 4))
    for index, classifier_info in enumerate(classifiers):
        confusion_matrixes[index,:] = np.array([confusion_matrix(y_test, np.round(classifier_info[0].predict(x_test))).ravel()])
    return confusion_matrixes * 100/total

confusion_matrixes = get_classifiers_cf(classifiers, x_test, y_test)


# In[18]:


def plot_metrics(dataframe, metric_indexes, indexes_results, orientation, x_label, y_label, classifiers):
    classifier_labels = [i[1] for i in classifiers]
    df_perf_results = pd.DataFrame(dataframe, columns=metric_indexes)
    df_perf_results.insert(0, 'classifier_type', classifier_labels, True)
    df_perf_results = pd.melt(df_perf_results, id_vars=['classifier_type'], value_vars=indexes_results, var_name='Metric')

    x = "value" if orientation=="h" else "classifier_type"
    y = "classifier_type" if orientation=="h" else "value"

    result_df = {'Metric': df_perf_results['Metric'],
                f'{x_label}': df_perf_results[x], f'{y_label}': df_perf_results[y] }

    fig = px.bar(result_df, x=x_label, y=y_label, color='Metric', barmode='group',
             height=400)
    fig.show()
# Generate dataset to plot using seaborn package
indexes = ["TN (%)", "FP (%)", "FN (%)", "TP (%)"]
indexes_result1 = ["TP (%)", "FP (%)", "FN (%)", "TN (%)"]

plot_metrics(confusion_matrixes, indexes, indexes_result1, "v", "Classifiers", "Percent", classifiers)


# In[19]:


df_confusion_matrixes = pd.DataFrame(confusion_matrixes, columns=indexes, index=[label for _, label in classifiers])
df_confusion_matrixes.T


# In[20]:


def perf_metrics(confusion_values):
    # [0] = TN, [1] = FP, [2] = FN, [3] = TP
    # 4.1 accuracy
    accuracy = (confusion_values[3] + confusion_values[0]) / (np.sum(confusion_values))
    # 4.2 precision
    precision = confusion_values[3] / (confusion_values[3] + confusion_values[1])
    # 4.3 specificity
    specificity = confusion_values[0] / (confusion_values[0] + confusion_values[1])
    # 4.4 TP rate
    tp_rate = confusion_values[3] / (confusion_values[3] + confusion_values[2])
    # 4.5 FP rate
    fp_rate = confusion_values[1] / (confusion_values[1] + confusion_values[0])
    # 4.6 NPV
    npv = confusion_values[0] / (confusion_values[0] + confusion_values[2])
    # 4.7 Rate of Misclassification
    misclassification_rate = (confusion_values[1] + confusion_values[2]) / (np.sum(confusion_values))
    # 4.8 F1 Score
    f1_score = 2*(precision * tp_rate) / (precision + tp_rate)

    return np.array([accuracy, precision, specificity, tp_rate, fp_rate, npv, misclassification_rate, f1_score])


def classifiers_perf_results(confusion_matrixes, classifiers, x_test, y_test):
    perf_results = np.zeros((confusion_matrixes.shape[0], 10))
    for i in np.arange(confusion_matrixes.shape[0]):
        perf_results[i,0:8] = perf_metrics(confusion_matrixes[i,:])

    # Calculate AUC and ROC curve and Mean Absolute Error (MEA)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mae = dict()

    for index, classifier_info in enumerate(classifiers):
        fpr[classifier_info[1]], tpr[classifier_info[1]], _ = roc_curve(y_test, classifier_info[0].predict(x_test))
        roc_auc[classifier_info[1]] = auc(fpr[classifier_info[1]], tpr[classifier_info[1]])
        mae[classifier_info[1]] = mean_absolute_error(y_test, classifier_info[0].predict(x_test))
    perf_results[:,8] = [roc_auc[i] for i in roc_auc]
    perf_results[:,9] = [mae[i] for i in mae]

    return perf_results


# In[21]:


perf_results = classifiers_perf_results(confusion_matrixes, classifiers, x_test, y_test)
# Metricas de accuracy, Precision, Specificity, Recall/TP rate
metric_indexes = ["CA", "Pre", "Spec", "Rec", "FPR", "NPV", "RMC", "F1", "AUC", "MAE"]

# Taxa de falso positivo, valor preditivo negativo, taxa de classificação incorreta e F1, respectivamente.
indexes_result2 = ["Pre", "Rec", "AUC", "MAE", "CA", "F1"]

plot_metrics(perf_results, metric_indexes, indexes_result2, "h", "No. of samples", "Classifiers", classifiers)


# ## Determinando o melhor modelo
# A partir das metricas resultantes de treino e teste dos modelos na base de dados desbalanceada é possível falar que os melhores modelos foram o naive bayes e LGBMClassifier. Onde LGBMClassifier apresentou a melhor acurácia, e precisão com o menor erro, e o modelo naive bayes com a melhores métricas de AUC e F1.
# 
# Antes de eleger o melhor modelo nessa primeira análise vamos balancear o datasets e realizar o treino e teste novamente dos nossos modelos.

# ## Balanceando o dataset
# Para balancear o dataset utilizaremos do `SMOTE`

# In[22]:


# Split dataset between train and test
seed = 10 # Set seed to get invariant results
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(entries, outputs, test_size=test_size, random_state=seed)
print('Train dataset shape:\nEntries: ', x_train.shape, '\nOutput: ', y_train.shape, '\n\n')
print('Test dataset shape:\nEntries: ', x_test.shape, '\nOutput: ', y_test.shape)


# In[23]:


# balanceando o dataset
over = SMOTE(sampling_strategy=0.1, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
x_train_balanced, y_train_balanced = pipeline.fit_resample(x_train, y_train)

print('Train dataset new shape:\nEntries: ', x_train_balanced.shape, '\nOutput: ', y_train_balanced.shape, '\n\n')


# Agora vamos testar novamente a proporção entre as classes de pagadores

# In[24]:


verify_unbalanced(pd.DataFrame({'test':y_train_balanced}), 'test')


# Dessa forma, agora os inadimplentes compõe 1/3 da base de dados. Com isso vamos treinar e testar os modelos novamente e verificar se há algum ganho do balanceamento ou não.

# In[25]:


balance_classifiers = get_classifiers()
k_fold_train(balance_classifiers ,x_train_balanced, y_train_balanced)


# In[26]:


classifiers_test(balance_classifiers, x_test, y_test)
confusion_matrixes = get_classifiers_cf(balance_classifiers, x_test, y_test)
plot_metrics(confusion_matrixes, indexes, indexes_result1, "v", "Classifiers", "Percent", balance_classifiers)


# In[27]:


df_confusion_matrixes = pd.DataFrame(confusion_matrixes, columns=indexes, index=[label+'(%)' for _, label in balance_classifiers])
df_confusion_matrixes.T


# In[28]:


perf_results = classifiers_perf_results(confusion_matrixes, balance_classifiers, x_test, y_test)
metric_indexes = ["CA", "Pre", "Spec", "Rec", "FPR", "NPV", "RMC", "F1", "AUC", "MAE"] # that stands for Classification accuracy, Precision, Specificity, Recall/TP rate, 
# False positive rate, negative predictive value, misclassification rate and F1, respectively.
indexes_result2 = ["Pre", "Rec", "AUC", "MAE", "CA", "F1"]

plot_metrics(perf_results, metric_indexes, indexes_result2, "h", "Value", "Classifiers", balance_classifiers)


# Com o balanceamento foi possível ver uma melhora na AUC de todos os modelos, e diferente de anteriormente  o LGBMClassifier representa o melhor modelo, com o menor erro e maior acurácia, AUC e F1 quando comparado-se com o Naive bayes que performou melhor com relação a essas métricas antes do balanceamento. Assim, agora vamos comparar o LGBMClassifier e o modelo de Naive bayes antes e depois do balancemento para determinar o melhor modelo:

# In[43]:


best_classifiers = ([classifiers[-2][0],'LGBMclass'], [classifiers[2][0],'Naivebayes'], [balance_classifiers[-2][0],'LGBMclass_balanced'], [balance_classifiers[2][0],'Naivebayes_balanced'])
classifiers_test(best_classifiers, x_test, y_test)
confusion_matrixes = get_classifiers_cf(best_classifiers, x_test, y_test)
# Generate dataset to plot using seaborn package
indexes = ["TN", "FP", "FN", "TP"]
indexes_result1 = ["TP", "FP", "FN", "TN"]

plot_metrics(confusion_matrixes, indexes, indexes_result1, "v", "Classifiers", "Percent", best_classifiers)


# In[44]:


perf_results = classifiers_perf_results(confusion_matrixes, best_classifiers, x_test, y_test)
plot_metrics(perf_results, metric_indexes, indexes_result2, "h", "Value", "Classifiers", best_classifiers)


# ## Melhor modelo
# Visualizando os resultados dos melhores modelos antes e depois do balanceamento fica claro que o balanceamento afetou positivamente a base de dados, aumentando consideravelmente a proporção de `True_positive` ou seja aumentou a probabilidade do modelo detectar um risco alto de inadimplencia e acusar `inadimplencia=1`. Além disso é possível notar que o modelo com a melhor performance , tendo em vista as metricas de AUC e F1, foi o LGBM com os dados balanceados.

# ## Treino e formulação dos resultados finais
# Agora vamos utilizar toda a base de dados contida no arquivo `treino.csv` para treino e validação utilizando várias folds, assim o modelo utilizará todos os dados de treinamento para convergir. Após isso será colocada a previsão do modelo na base de dados de teste contida no arquivo `teste.csv`.

# In[30]:


# Separate entries from outputs
dataset = df.to_numpy() # Converting from Pandas dataframe to Numpy
entries = df.loc[:, df.columns != 'inadimplente'].to_numpy(dtype=np.float64)
outputs = df['inadimplente'].to_numpy(dtype=np.int64)


# In[31]:


# transformar o data set
over = SMOTE(sampling_strategy=0.1, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
x_train_balanced, y_train_balanced = pipeline.fit_resample(entries, outputs)
print('Dataset old shape:\nEntries: ', entries.shape, '\nOutput: ', outputs.shape, '\n\n')
print('Train dataset new shape:\nEntries: ', x_train_balanced.shape, '\nOutput: ', y_train_balanced.shape, '\n\n')


# In[32]:


# Pegando somente o LGBMCLassifier
final_model = [get_classifiers()[-2]]


# In[33]:


k_fold_train(final_model ,x_train_balanced, y_train_balanced)


# Agora vamos pegar as entradas da base de dados no arquivo `teste.csv` para produzir a previsão do modelo.

# In[34]:


test_df = pd.read_csv('teste.csv')
test_df = test_df.apply(lambda x: outlier_capping(x))
test_df = test_df.apply(lambda x:  missing_val_treat(x))

test_x_train = test_df.to_numpy()


# In[35]:


inadimplente_test =  final_model[0][0].predict(test_x_train)
inadimplente_test_prob = final_model[0][0].predict_proba(test_x_train)[:, 1]

test_df['inadimplente'] = inadimplente_test
test_df['inadimplente_prob'] = inadimplente_test_prob


# In[36]:


test_df.head()


# In[37]:


test_df.to_csv('rodrigo_reults.csv')

