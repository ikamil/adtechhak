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

SOURCEFILE = r'/mnt/c/temp/hakaton/train.csv'
TRAINFILE = r'/mnt/c/temp/hakaton/train_data.csv'
PREDFILE = r'/mnt/c/temp/hakaton/pred_data.csv'
RESFILE = r'/mnt/c/temp/hakaton/result_data.csv'
INDEXCOL = 'id'
RESULTCOL = 'segment'

engine = create_engine('postgresql://postgres:qwe123q@localhost:65432/hakaton', encoding='utf-8')

if False: #выполняется однократно вначале
    with engine.connect() as con:
        con.execute("""drop table if exists train_raw;""")
        con.execute("""create table train_raw (segment int,gamecategory text,subgamecategory text, bundle text,created timestamp,shift text,oblast text,city text,os text,osv text);""")
        con.execute(f"""copy train_raw from '{SOURCEFILE}' csv header;""")
        con.execute("""drop table if exists train;""")
        con.execute("""create table train (id serial, segment int,gamecategory text,subgamecategory text, bundle text,created timestamp,shift text,oblast text,city text,os text,osv text);""")
        con.execute("""insert into train(segment,gamecategory,subgamecategory, bundle,created,shift,oblast,city,os,osv) select segment,gamecategory,subgamecategory, bundle,created,shift,oblast,city,os,osv from train_raw;""")


if True:
    print('===  Preparing data using SQL ===')
    with engine.connect() as con:
        con.execute("""drop table if exists all_data""")
        con.execute("""create table all_data as
with sta as (
    select id, segment
         , count(1) over (partition by lower(coalesce(gamecategory, '')))                   gamecat
         , count(1) over (partition by lower(coalesce(subgamecategory, '')))                game
         , tsvector_to_array(to_tsvector(
            replace(regexp_replace(
                            regexp_replace(
                                    replace(bundle, 'com.', '')
                                , '([A-Z])', '.\1', 'g')
                        , '\.{2,}', '.', 'g'), '.', ' ')
        ))                                                                                  app
         , extract(hour from created)                                                       hourc
         , extract(dow from created)                                                        dowc
         , extract(month from created)                                                      monthc
         , count(1) over (partition by lower(coalesce(city, oblast, '')))                   geo
         , count(1) over (partition by lower(coalesce(os, '') || '|' || coalesce(osv, ''))) dev
    from train
)
select id, gamecat, game, hourc, dowc, monthc, geo
        , count(1) over (partition by app[1]) app1
        , count(1) over (partition by coalesce(app[2], app[1])) app2
        , count(1) over (partition by coalesce(app[3], app[2], app[1])) app3
        , count(1) over (partition by coalesce(app[4], app[3], app[2], app[1])) app4
        , segment
       from sta
           """ )
        con.execute("""drop table if exists train_data""")
        con.execute("""create table train_data as select * from all_data tablesample system(90) repeatable (1)""")
        con.execute("""drop table if exists test_data""")
        con.execute("""create table test_data as select * from all_data tablesample system(10) repeatable (2)""")
        con.execute("""drop table if exists test_result""")
        con.execute("""create table test_result as select * from train t where exists (select 1 from test_data d where d.id=t.id);""")
        con.execute("copy train_data to '%s' (format csv, header true)" % TRAINFILE)
        con.execute("copy test_data to '%s' (format csv, header true)" % PREDFILE)
        con.execute("copy test_result to '%s' (format csv, header true)" % RESFILE)
        print('=== Finished SQL Data prepare ===')



alldata = pd.read_csv(TRAINFILE, index_col=INDEXCOL)
cols = list(alldata.columns)
train_cols = cols[:-1]
test_vals = [14,1427,10,5,7,31,1,1,1,1]
clft = RandomForestClassifier(max_depth=23) #0.6521212121212121
clfa = DecisionTreeClassifier(max_depth=None, min_samples_leaf=51, criterion='entropy') #0.6423376623376623

class_col = cols[-1]

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

resdata = pd.read_csv(RESFILE, index_col=INDEXCOL)
preddata[RESULTCOL] = pred
# result = preddata[RESULTCOL] if False else preddata

print('****  RESULTDT:  %s  *****' % preddata)
print('****  RESULTDA:  %s  *****' % resdata)

result = resdata.join(preddata[[RESULTCOL]], rsuffix='_predict')
print('****  RESULT:  %s  *****' % result)

nm = '_%s_%s' % (str(clf.__class__()), round(max(scoret, scorea), 4))
result.to_csv(RESFILE.replace('_res', nm))
