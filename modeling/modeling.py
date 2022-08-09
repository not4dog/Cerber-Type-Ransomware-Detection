from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# 분류알고리즘 : SVC, RF, NB
class Classifiers():

    def __init__(self, X, Y):

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        def do_svm(self):
            clf = SVC()
            clf.fit(self.x_train, self.y_train)
            y_pred = clf.predict(self.x_test)
            return accuracy_score(self.y_test, y_pred)
            data = f.read()

        def do_randomforest(self, mode):
            clf = RandomForestClassifier()
            clf.fit(self.x_train, self.y_train)
            
            if mode == 1:
                return clf.feature_importances_

            y_pred = clf.predict(self.x_test)
            return accuracy_score(self.y_test, y_pred)
        
        def do_naivebayes(self):
            clf = GaussianNB()
            clf.fit(self.x_train, self.y_train)
            y_pred = clf.predict(self.x_test)

		return accuracy_score(self.y_test, y_pred)

# 탐지률 저장부분 
def do_all(self):
    
    rns = []

    rns.append(self.do_svm())
    rns.append(self.do_randomforest(0))
    rns.append(self.do_naivebayes())

    return rns
