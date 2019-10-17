# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here
df = pd.read_json(path, lines=True)
df.columns = [x.replace(" ","_") for x in df.columns ]
missing_data = df.isnull()
df.drop(columns=['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'], inplace=True)
X = df.drop(columns="fit")
y=df.fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=6)

# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here
g_by_category = df.groupby(by="category")
cat_fit = g_by_category.count()
cat_fit = cat_fit.unstack()
plot_barh(cat_fit, "Frequency")

# Code ends here


# --------------
# Code starts here
cat_len = g_by_category['length'].value_counts().unstack()
plot_barh(cat_len, "Length")

# Code ends here


# --------------
# Code starts here
def get_cms(x):
    if(type(x)==str):
        if(len(x)>3):
            x = (int(x[0])*30.48) + (int(x[4:-2])*2.54)
        else:
            x = (int(x[0])*30.48)
    return x
X_train.height = X_train['height'].apply(get_cms)
X_test.height = X_test['height'].apply(get_cms)

# Code ends here


# --------------
# Code starts here
X_train = X_train.dropna(subset=['height', 'length', 'quality'])
X_test = X_test.dropna(subset=['height', 'length', 'quality'])
y_train = y_train[[x in X_train.index for x in y_train.index]]
y_test = y_test[[x in X_test.index for x in y_test.index]]
X_train.bra_size = X_train.bra_size.fillna(X_train.bra_size.mean())
X_test.bra_size = X_test.bra_size.fillna(X_test.bra_size.mean())
X_train.bra_size = X_train.bra_size.fillna(X_train.bra_size.mean())
X_test.bra_size = X_test.bra_size.fillna(X_test.bra_size.mean())
mode_1 = X_train.cup_size.mode()[0]
mode_2 = X_test.cup_size.mode()[0]
X_train.cup_size = X_train.cup_size.fillna(mode_1)
X_test.cup_size = X_test.cup_size.fillna(mode_2)

# Code ends here


# --------------
# Code starts here
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# Code starts here
model = DecisionTreeClassifier(random_state=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)
precision = precision_score(y_test, y_pred, average=None)
print(precision)

# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here
model = DecisionTreeClassifier(random_state=6)
grid = GridSearchCV(estimator=model, param_grid=parameters)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Code ends here


