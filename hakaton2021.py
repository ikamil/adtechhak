from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

TRAINFILE = r'/mnt/c/temp/hakaton/train_data.csv'
PREDFILE = r'/mnt/c/temp/hakaton/pred_data.csv'
RESFILE = r'/mnt/c/temp/hakaton/result_data.csv'
INDEXCOL = 'segment'
RESULTCOL = 'segment'

engine = create_engine('postgresql://postgres:qwe123q@localhost:65432/hakaton', encoding='utf-8')

if True:
    print('===  Preparing data using SQL ===')
    with engine.connect() as con:
        con.execute("""drop table if exists all_data""")
        con.execute("""create table all_data as
    select distinct segment, lower(coalesce(gamecategory,'')||'|'||coalesce(subgamecategory,'')) game
    , lower(array_to_string(tsvector_to_array(to_tsvector(
        replace(regexp_replace(
        regexp_replace(
        replace(bundle,'com.','')
        ,'([A-Z])','.\1','g')
        ,'\.{2,}','.','g'),'.',' ')
    )),'|')) app
    , extract(hour from created)||'|'||extract(dow from created)||'|'||extract(month from created) date
    , lower(coalesce(city,oblast,'')) geo
    , lower(coalesce(os,'')||'|'||coalesce(osv,'')) dev
    from train
           """ )
        con.execute("""drop table if exists train_data""")
        con.execute("""create table train_data as select * from all_data tablesample system(90) repeatable (1)""")
        con.execute("""drop table if exists test_data""")
        con.execute("""create table test_data as select * from all_data tablesample system(10) repeatable (2)""")
        con.execute("copy train_data to '%s' (format csv, header true)" % TRAINFILE)
        con.execute("copy test_data to '%s' (format csv, header true)" % PREDFILE)
        print('=== Finished SQL Data prepare ===')



alldata = pd.read_csv(TRAINFILE, index_col=INDEXCOL)
cols = list(alldata.columns)
train_cols = ['game', 'app', 'date', 'geo', 'dev']
test_vals = ['games|word','fugo|wow','19|2|7','нальчик','android|11.0']
clft = RandomForestClassifier(max_depth=22)
clfa = DecisionTreeClassifier(max_depth=None, min_samples_leaf=55, criterion='entropy')
class_col = cols[-2]

X = alldata.loc[:, train_cols]
print('=== TRAIN DATASET: %s  =====\n\n' % X)
y = alldata[class_col].to_numpy()
# print('=== TRAIN CLASSES: %s  =====\n\n' % y)

preddata = pd.read_csv(PREDFILE, index_col=INDEXCOL)
X_pred = preddata.loc[:, train_cols]
print('---- Predicting Dataset: %s -----\n\n' % X_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=X_pred.__len__())
print('=== Start Train Meth1 for %s ===' % X_train.__len__())
clft.fit(X_train, y_train)
print('=== Finished  ==== \n\n')
print('=== Start Train Meth2 for %s ===' % X_train.__len__())
clfa.fit(X_train, y_train)
print('=== Finished  ==== \n\n')
print('=== Start Predict Score for Meth1 ===')
predt = clft.predict(X_test)
scoret = accuracy_score(predt, y_test)
print('=== Start Predict Score for Meth2 ===')
preda = clfa.predict(X_test)
scorea = accuracy_score(preda, y_test)
print('!!!---- got self Score with Methods: %s and %s \n for %s columns and %s records, samples: %s and %s ----==== !!!\n\n' % (scoret, scorea, train_cols, X_train.__len__(), clft.predict([test_vals]), clfa.predict([test_vals])))

clf = clfa if scorea > scoret else clft
pred = clf.predict(X_pred)

preddata[RESULTCOL] = pred
result = preddata[RESULTCOL]
print('****  RESULT:  %s  *****' % result)
nm = '_%s_%s' % (str(clf.__class__()), round(max(scoret, scorea), 4))
result.to_csv(RESFILE.replace('_res', nm))
