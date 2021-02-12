### 2.2 Travail à faire

1. Charger et transformer les données de façon à ce qu’elles soient reconnues comme des transactions. En pratique on construit un tableau de données binaires.  <br/>
Les instructions Python suivantes permettent de charger le jeu de données retail :

```python
import pandas as pd
db = pd.read_csv('retail_dataset.csv',sep=',',header=0)
print(db.head(10))
```

2. Utiliser l’algorithme Apriori pour extraire les itemsets fréquents et les maximaux. Vous choisirez un support minimum de 3%.  <br/>
— Que se passe t’il si on fait varier le seuil du support ?  <br/>
— Tracer une courbe montrant l’évolution du nombre de itemsets extraits en fonction du support minimum.  <br/>

```python
import numpy as np
db_trans = [[val if type(val) == str else 'missing_value' 
            for val in transaction] 
                for transaction in db.values]

TDB = TransactionEncoder()
TDBA = TDB.fit(db_trans).transform(db_trans)
#print(TDB.columns_)
#print(TDBA)

dbf = pd.DataFrame(TDBA, columns=TDB.columns_).drop(columns='missing_value')
#print(dbf)

DBI = {'support': 0.3, 'results' :apriori(dbf,min_support=0.3, use_colnames=True)}
print(DBI['results'])
```

```python
## Que se passe t’il si on fait varier le seuil du support ?
DBIS = [{'support': (i/2)/10, 
        'results' :apriori(dbf,min_support=(i/2)/10, use_colnames=True)} 
            for i in reversed(range(1,11)) ]
print(pd.DataFrame(data= {
    'suport': [x['support'] for x in DBIS],
    'nb_of_itemsets': [len(x['results']) for x in DBIS]}))
```
```python
## Tracer une courbe montrant l’évolution du nombre de itemsets extraits 
## en fonction du support minimum.
dataset2 = pd.DataFrame(data= {
    'min_support': [x['support'] for x in DBIS ],
    'nb_of_itemsets': [len(x['results']) for x in DBIS ]})

import seaborn as sns
sns.lineplot(data=dataset2 ,x='min_support', y="nb_of_itemsets")
```

3. Nous souhaitons pouvoir filtrer les itemsets selon la présence d’items ou d’un ensemble
d’items.  <br/>
Par exemple, quels sont les itemsets qui contiennent le produit ’Eggs’ ? les produits
{’Eggs’,’Meat’} ?

> ##### Note
> Plusieurs solutions s‘offrent à vous pour la recherche d’itemsets répondant à des conditions de présence d’items.  <br/>
> Vous pouvez par exemple utiliser les opérateurs de comparaison de [pandas.Series](https://pandas.pydata.org/pandas-docs/stable/reference/series.html).

```python
def val_to_column_name(itemsets: pd.DataFrame):
    col_names = itemsets.columns.tolist()
    transactions = itemsets.values.tolist()
    
    return [[col_name
            for val, col_name in zip(transaction, col_names) if val != False]
                for transaction in transactions]

def filter_itemsets(products: list, itemsets: pd.DataFrame):
    query = '&'.join([f'{column}==True' for column in set(products)])
    res = itemsets.query(query)

    return res, val_to_column_name(res)

print(filter_itemsets(['Eggs'], dbf)[1])
print(filter_itemsets(['Eggs', 'Meat'], dbf)[1])
```

4. Utiliser l’algorithme Apriori pour extraire les règles d’association à partir des itemsets
fréquents et des itemsets maximaux. Vous choisirez une confiance minimale de 75%. Extraire
les règles ayant pour conséquents ’Chesse’.
   
```python
DFAP = apriori(dbf, min_support=0.1, use_colnames=True)
#print(DFAP)

DFAPAR = association_rules(DFAP,metric="confidence",min_threshold=0.75)
#print(DFAPAR)

DFAPAR = DFAPAR[DFAPAR['consequents'] == {'Cheese'}][['support', 'lift', 'confidence']]
print(DFAPAR)
```

> ##### Note
> La fonction association_rules() renvoie un objet de type pandas.dataframe contenant
les différentes règles d’association, chacune décrite par différentes caractéristiques qui sont
l’antécédent, le conséquent, et 7 indicateurs numériques d’évaluation des règles.  <br/>
> Il faudra adapter l’affichage pour disposer que des informations liées aux mesures support, lift et la confidence.

5. Compléter l’analyse des différentes règles d’association extraites via des graphiques permet-
tant d’étudier la corrélation entre les trois mesures (lift, confiance et support) d’évaluation
des règles.
   
```python
sns.pairplot(data = DFARF)
```
