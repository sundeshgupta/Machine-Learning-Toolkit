#!/usr/bin/python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from utils import plot_classification_report, plot_confusion_matrix
import time

import cgi, cgitb

import matplotlib.style
import matplotlib

matplotlib.style.use('default')

# Create instance of FieldStorage
form = cgi.FieldStorage()

class Configuration:
    algorithm = "algorithm"
    data = "data"
    topic = "topic"

    def __str__ (self):
        return "Algorothm: " + str(self.algorithm) + " Data: " + str(self.data) + " Topic: " + str(self.topic)

class Data:
    def __init__(self):
        self.start_time = time.time()

        if (Configuration.data == "Social_Network_Ads.csv"):
            self.dataset = pd.read_csv(str(Configuration.data))

        if (Configuration.algorithm == "linear_regression"):
            self.X = self.dataset.iloc[:, :-1].values
            self.y = self.dataset.iloc[:, 1].values
        elif (Configuration.algorithm == "logistic_regression" or Configuration.algorithm == "svc"
                or Configuration.algorithm == "decision_tree_classification" or Configuration.algorithm == "random_forest_classification" or
                    Configuration.algorithm == "knn"):
            if (Configuration.data=="Social_Network_Ads.csv"):
                self.X = self.dataset.iloc[:, [2,3]].values
                self.y = self.dataset.iloc[:, 4].values
            else:
                if (Configuration.data == "moons"):
                    from sklearn.datasets.samples_generator import make_moons
                    self.X, self.y = make_moons(100, noise=.2, random_state = 0)
                elif (Configuration.data == "circles"):
                    from sklearn.datasets.samples_generator import make_circles
                    self.X, self.y = make_circles(100, factor=.5, noise=.1, random_state = 0)
        elif (Configuration.algorithm == "polynomial_regression"):
            self.X = self.dataset.iloc[:, 1:2].values
            self.y = self.dataset.iloc[:, 2].values
        elif (Configuration.algorithm == "kmeans"):
            self.X = self.dataset.iloc[:, [3, 4]].values
            self.y = None

        if (Configuration.data == "Social_Network_Ads.csv"):
            self.directory = "SocialNetworkAds"
        elif (Configuration.data == "moons"):
            self.directory = "Moons"
        elif (Configuration.data == "circles"):
            self.directory = "Circles"

    def input(self, algorithm, pdegree = int(2), t_size=20):
        def linearregression():
            self.test_size = t_size/100
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = 0)
            self.y_pred = np.zeros(np.shape(self.y_test))
        def polynomialregression():
            self.pdegree = pdegree

        def logisticregression():
            self.test_size=t_size/100
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size)
            self.y_pred = np.zeros(np.shape(self.y_test))

        def knn():
            self.test_size=t_size/100
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size)
            self.y_pred = np.zeros(np.shape(self.y_test))

        def svc():
            self.test_size=t_size/100
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size)
            self.y_pred = np.zeros(np.shape(self.y_test))

        def decisiontreeclassification():
            self.test_size=t_size/100
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size)
            self.y_pred = np.zeros(np.shape(self.y_test))

        def randomforestclassification():
            self.test_size=t_size/100
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size)
            self.y_pred = np.zeros(np.shape(self.y_test))

        def kmeans():
            pass

        func, arg = {
            "polynomial_regression": (polynomialregression, ()),
            "linear_regression" : (linearregression, ()),
            "logistic_regression" : (logisticregression, ()),
            "knn" : (knn, ()),
            "svc" : (svc, ()),
            "decision_tree_classification" : (decisiontreeclassification, ()),
            "random_forest_classification" : (randomforestclassification, ()),
            "kmeans" : (kmeans, ()),
        }.get(algorithm, "failed to find the output data of chosen algorithm")

        return func(*arg)

    def preprocess(self,algorithm):
        def linearregression():
            pass
        def polynomialregression():
            pass
        def logisticregression():
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.transform(self.X_test)
        def knn():
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.transform(self.X_test)
        def svc():
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.transform(self.X_test)
        def decisiontreeclassification():
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.transform(self.X_test)
        def randomforestclassification():
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.transform(self.X_test)
        def kmeans():
            pass
        func, arg = {
            "polynomial_regression": (polynomialregression, ()),
            "linear_regression" : (linearregression, ()),
            "logistic_regression" : (logisticregression, ()),
            "knn" : (knn, ()),
            "svc" : (svc, ()),
            "decision_tree_classification" : (decisiontreeclassification, ()),
            "random_forest_classification" : (randomforestclassification, ()),
            "kmeans" : (kmeans, ()),
        }.get(algorithm, "failed to find the output data of chosen algorithm")
        return func(*arg)

    def output(self, algorithm, fitted_object, kernel = None):
        def polynomialregression():
            plt.scatter(self.X, self.y, color = 'blue')
            plt.plot(self.X, fitted_object[0].predict(fitted_object[1].fit_transform(self.X)), color = 'red')
            plt.title('(Polynomial Regression)')
            plt.savefig('./PolynomialRegression/PositionSalaries/train_'+str(self.pdegree)+'.svg', format='svg', dpi=1200)
            plt.show()
            from sklearn.metrics import mean_squared_error
            print(mean_squared_error(self.y, fitted_object[0].predict(fitted_object[1].fit_transform(self.X))))
        def linearregression():
            self.y_pred = fitted_object.predict((self.X_test))
            plt.scatter(self.X_test, self.y_test, color = 'blue')
            plt.plot(self.X_train, fitted_object.predict((self.X_train)), color = 'red')
            plt.title('Test set')
            plt.savefig('./LinearRegression/SalaryData/test.svg', format='svg', dpi=1200)
            plt.show()
            plt.close()
            plt.scatter(self.X_train,self.y_train, color = 'blue')
            plt.plot(self.X_train, fitted_object.predict((self.X_train)), color = 'red')
            plt.title('Training set')
            plt.savefig('./LinearRegression/SalaryData/train.svg', format='svg', dpi=1200)
            plt.show()
            plt.close()
            from sklearn.metrics import mean_squared_error
            print(mean_squared_error(self.y_test, self.y_pred))
        def logisticregression():
            self.y_pred = fitted_object.predict(self.X_test)
            print(fitted_object.coef_)
            print(fitted_object.intercept_)
            plt.grid(False)
            from sklearn.metrics import accuracy_score
            print(accuracy_score(self.y_test, self.y_pred))
            from matplotlib.colors import ListedColormap
            from matplotlib import colors
            cmap = colors.LinearSegmentedColormap(
            'red_blue_classes',
            {'red': [(0, 1, 1), (1, 0.7, 0.7)],
            'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
            'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
            plt.cm.register_cmap(cmap=cmap)

            X_set, y_set = self.X_train, self.y_train
            cm=plt.cm.RdBu
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000', '#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')
            plt.title('Training Set')
            plt.legend()
            plt.savefig('./LogisticRegression/'+self.directory+'/train.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()
            # Visualising the Test set results
            from matplotlib.colors import ListedColormap
            X_set, y_set = self.X_test, self.y_test
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000','#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')
            plt.title('Test Set')
            plt.legend()
            plt.grid(False)
            plt.savefig('./LogisticRegression/'+self.directory+'/test.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()

            from sklearn.metrics import confusion_matrix
            cm_plt=plot_confusion_matrix(confusion_matrix(self.y_test, self.y_pred))
            cm_plt.savefig('./LogisticRegression/'+self.directory+'/cm.svg', format='svg', dpi=1200)
            # cm_plt.show()
            cm_plt.close()

            cr_plt=plot_classification_report(self.y_test, self.y_pred)
            cr_plt.savefig('./LogisticRegression/'+self.directory+'/cr.svg', format='svg', dpi=1200)
            # cr_plt.show()
            cr_plt.close()

            with open("./LogisticRegression/"+self.directory+"/coef.txt", "w") as text_file:
                print(fitted_object.coef_, file=text_file)

            from sklearn.metrics import accuracy_score
            with open("./LogisticRegression/"+self.directory+"/intercept.txt", "w") as text_file:
                print(fitted_object.intercept_, file=text_file)

            with open("./LogisticRegression/"+self.directory+"/time.txt", "w") as text_file:
                print(str(time.time()-self.start_time) + " seconds", file=text_file)

            from sklearn.metrics import accuracy_score
            with open("./LogisticRegression/"+self.directory+"/accu.txt", "w") as text_file:
                print(accuracy_score(self.y_test, self.y_pred), file=text_file)
        def knn():
            self.y_pred = fitted_object.predict(self.X_test)
            # print(fitted_object.coef_)
            # print(fitted_object.intercept_)
            plt.grid(False)
            from matplotlib.colors import ListedColormap
            from matplotlib import colors
            cmap = colors.LinearSegmentedColormap(
            'red_blue_classes',
            {'red': [(0, 1, 1), (1, 0.7, 0.7)],
            'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
            'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
            plt.cm.register_cmap(cmap=cmap)

            X_set, y_set = self.X_train, self.y_train
            cm=plt.cm.RdBu
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000', '#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')
            plt.title('Training Set')
            plt.legend()
            plt.savefig('./knn/'+self.directory+'/train.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()
            # Visualising the Test set results
            from matplotlib.colors import ListedColormap
            X_set, y_set = self.X_test, self.y_test
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000','#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')
            plt.title('Test Set')
            plt.legend()
            plt.grid(False)
            plt.savefig('./test.svg', format= 'svg', dpi=1200)
            plt.savefig('./knn/'+self.directory+'/test.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()

            from sklearn.metrics import confusion_matrix
            cm_plt=plot_confusion_matrix(confusion_matrix(self.y_test, self.y_pred))
            cm_plt.savefig('./knn/'+self.directory+'/cm.svg', format='svg', dpi=1200)
            # cm_plt.show()
            cm_plt.close()

            cr_plt=plot_classification_report(self.y_test, self.y_pred)
            cr_plt.savefig('./knn/'+self.directory+'/cr.svg', format='svg', dpi=1200)
            # cr_plt.show()
            cr_plt.close()

            with open("./knn/"+self.directory+"/time.txt", "w") as text_file:
                print(str(time.time()-self.start_time) + " seconds", file=text_file)

            from sklearn.metrics import accuracy_score
            with open("./knn/"+self.directory+"/accu.txt", "w") as text_file:
                print(accuracy_score(self.y_test, self.y_pred), file=text_file)
        def svc():
            self.y_pred = fitted_object.predict(self.X_test)
            # print(fitted_object.coef_)
            # print(fitted_object.intercept_)
            plt.grid(False)
            from matplotlib.colors import ListedColormap
            from matplotlib import colors
            cmap = colors.LinearSegmentedColormap(
            'red_blue_classes',
            {'red': [(0, 1, 1), (1, 0.7, 0.7)],
            'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
            'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
            plt.cm.register_cmap(cmap=cmap)

            X_set, y_set = self.X_train, self.y_train
            cm=plt.cm.RdBu
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000', '#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')

            # plt.scatter(fitted_object.support_vectors_[:, 0], fitted_object.support_vectors_[:, 1],s=300, linewidth=1, facecolors='none', edgecolors='k')
            # print(np.shape(fitted_object.support_vectors_))
            #Plot the decision function
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # create grid to evaluate model
            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = fitted_object.decision_function(xy).reshape(XX.shape)
            plt.contour(XX, YY, Z, colors='k', levels=[-1,1], alpha=0.5,
                       linestyles=['--', '--'])

            plt.title('Training Set')
            plt.legend()
            plt.savefig('./svc/'+self.directory+'/train.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()
            # Visualising the Test set results
            from matplotlib.colors import ListedColormap
            X_set, y_set = self.X_test, self.y_test
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000','#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')

            # plt.scatter(fitted_object.support_vectors_[:, 0], fitted_object.support_vectors_[:, 1],s=300, linewidth=1, facecolors='none', edgecolors='k')

            plt.title('Test Set')
            plt.legend()
            plt.grid(False)
            plt.savefig('./test.svg', format= 'svg', dpi=1200)
            plt.savefig('./svc/'+self.directory+'/test.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()

            from sklearn.metrics import confusion_matrix
            cm_plt=plot_confusion_matrix(confusion_matrix(self.y_test, self.y_pred))
            cm_plt.savefig('./svc/'+self.directory+'/cm.svg', format='svg', dpi=1200)
            # cm_plt.show()
            cm_plt.close()

            cr_plt=plot_classification_report(self.y_test, self.y_pred)
            cr_plt.savefig('./svc/'+self.directory+'/cr.svg', format='svg', dpi=1200)
            # cr_plt.show()
            cr_plt.close()

            with open("./svc/"+self.directory+"/time.txt", "w") as text_file:
                print(str(time.time()-self.start_time) + " seconds", file=text_file)

            from sklearn.metrics import accuracy_score
            with open("./svc/"+self.directory+"/accu.txt", "w") as text_file:
                print(accuracy_score(self.y_test, self.y_pred), file=text_file)
            with open("./svc/"+self.directory+"/intercept.txt", "w") as text_file:
                if (kernel == 'linear'):
                    print(fitted_object.intercept_, file=text_file)
                else:
                    print("Intercept is only available when using a linear kernel", file=text_file)
            with open("./svc/"+self.directory+"/coef.txt", "w") as text_file:
                if (kernel == 'linear'):
                    print(fitted_object.intercept_, file=text_file)
                else:
                    print("Coefficients are only available when using a linear kernel", file=text_file)


            from mpl_toolkits import mplot3d
            fig = plt.figure()
            ax = fig.add_subplot(221, projection='3d')
            ax.scatter3D(self.X_test[:, 0], self.X_test[:, 1], fitted_object.decision_function(self.X_test), c = self.y_pred, cmap=ListedColormap(['#FF0000','#0000FF']))
            ax.view_init(elev=30, azim=-145)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            # plt.savefig("front.svg", format = "svg", dpi=1200)
            # plt.show()
            ax = fig.add_subplot(222, projection='3d')
            ax.scatter3D(self.X_test[:, 0], self.X_test[:, 1], fitted_object.decision_function(self.X_test), c = self.y_pred, cmap=ListedColormap(['#FF0000','#0000FF']))
            ax.view_init(elev=89 ,azim=-91)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            ax = fig.add_subplot(223, projection='3d')
            ax.scatter3D(self.X_test[:, 0], self.X_test[:, 1], fitted_object.decision_function(self.X_test), c = self.y_pred, cmap=ListedColormap(['#FF0000','#0000FF']))
            ax.view_init(elev=0, azim=-179)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            ax = fig.add_subplot(224, projection='3d')
            ax.scatter3D(self.X_test[:, 0], self.X_test[:, 1], fitted_object.decision_function(self.X_test), c = self.y_pred, cmap=ListedColormap(['#FF0000','#0000FF']))
            ax.view_init(elev=0, azim=-91)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            plt.savefig("./svc/"+self.directory+"/3d.svg", format = "svg", dpi=1200)
            # plt.show()
            plt.close()
        def decisiontreeclassification():
            self.y_pred = fitted_object.predict(self.X_test)
            # print(fitted_object.coef_)
            # print(fitted_object.intercept_)
            plt.grid(False)
            from matplotlib.colors import ListedColormap
            from matplotlib import colors
            cmap = colors.LinearSegmentedColormap(
            'red_blue_classes',
            {'red': [(0, 1, 1), (1, 0.7, 0.7)],
            'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
            'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
            plt.cm.register_cmap(cmap=cmap)

            X_set, y_set = self.X_train, self.y_train
            cm=plt.cm.RdBu
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000', '#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')


            plt.title('Training Set')
            plt.legend()
            plt.savefig('./DecisionTreeClassification/'+self.directory+'/train.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()
            # Visualising the Test set results
            from matplotlib.colors import ListedColormap
            X_set, y_set = self.X_test, self.y_test
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000','#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')

            plt.title('Test Set')
            plt.legend()
            plt.grid(False)
            plt.savefig('./test.svg', format= 'svg', dpi=1200)
            plt.savefig('./DecisionTreeClassification/'+self.directory+'/test.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()

            from sklearn.metrics import confusion_matrix
            cm_plt=plot_confusion_matrix(confusion_matrix(self.y_test, self.y_pred))
            cm_plt.savefig('./DecisionTreeClassification/'+self.directory+'/cm.svg', format='svg', dpi=1200)
            # cm_plt.show()
            cm_plt.close()

            cr_plt=plot_classification_report(self.y_test, self.y_pred)
            cr_plt.savefig('./DecisionTreeClassification/'+self.directory+'/cr.svg', format='svg', dpi=1200)
            # cr_plt.show()
            cr_plt.close()

            with open("./DecisionTreeClassification/"+self.directory+"/time.txt", "w") as text_file:
                print(str(time.time()-self.start_time) + " seconds", file=text_file)

            from sklearn.metrics import accuracy_score
            with open("./DecisionTreeClassification/"+self.directory+"/accu.txt", "w") as text_file:
                print(accuracy_score(self.y_test, self.y_pred), file=text_file)

            from sklearn.tree import export_graphviz
            import graphviz
            export_graphviz(fitted_object, out_file="./DecisionTreeClassification/"+self.directory+"/mytree.dot")
            with open("./DecisionTreeClassification/"+self.directory+"/mytree.dot") as f:
                dot_graph = f.read()
            graphviz.Source(dot_graph)
            from subprocess import check_call
            check_call(['dot','-Tpng','./DecisionTreeClassification/'+self.directory+'/mytree.dot','-o','./DecisionTreeClassification/'+self.directory+'/tree.png'])
        def randomforestclassification():
            self.y_pred = fitted_object.predict(self.X_test)
            # print(fitted_object.coef_)
            # print(fitted_object.intercept_)
            plt.grid(False)
            from matplotlib.colors import ListedColormap
            from matplotlib import colors
            cmap = colors.LinearSegmentedColormap(
            'red_blue_classes',
            {'red': [(0, 1, 1), (1, 0.7, 0.7)],
            'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
            'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
            plt.cm.register_cmap(cmap=cmap)

            X_set, y_set = self.X_train, self.y_train
            cm=plt.cm.RdBu
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000', '#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')


            plt.title('Training Set')
            plt.legend()
            plt.savefig('./DecisionTreeClassification/'+self.directory+'/train.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()
            # Visualising the Test set results
            from matplotlib.colors import ListedColormap
            X_set, y_set = self.X_test, self.y_test
            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
            plt.contourf(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.5, cmap = 'red_blue_classes')
            plt.contour(X1, X2, fitted_object.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), [0.5], linewidths = 1,  colors='white')
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(['#FF0000','#0000FF'])(i), label = j, alpha = 0.8, edgecolors='k')

            plt.title('Test Set')
            plt.legend()
            plt.grid(False)
            plt.savefig('./test.svg', format= 'svg', dpi=1200)
            plt.savefig('./DecisionTreeClassification/'+self.directory+'/test.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()

            from sklearn.metrics import confusion_matrix
            cm_plt=plot_confusion_matrix(confusion_matrix(self.y_test, self.y_pred))
            cm_plt.savefig('./DecisionTreeClassification/'+self.directory+'/cm.svg', format='svg', dpi=1200)
            # cm_plt.show()
            cm_plt.close()

            cr_plt=plot_classification_report(self.y_test, self.y_pred)
            cr_plt.savefig('./DecisionTreeClassification/'+self.directory+'/cr.svg', format='svg', dpi=1200)
            # cr_plt.show()
            cr_plt.close()

            with open("./DecisionTreeClassification/"+self.directory+"/time.txt", "w") as text_file:
                print(str(time.time()-self.start_time) + " seconds", file=text_file)

            from sklearn.metrics import accuracy_score
            with open("./DecisionTreeClassification/"+self.directory+"/accu.txt", "w") as text_file:
                print(accuracy_score(self.y_test, self.y_pred), file=text_file)
        def kmeans():

            self.y = fitted_object.fit_predict(self.X)
            plt.scatter(self.X[:,0], self.X[:,1], c = self.y, cmap = 'rainbow', edgecolors = 'k')
            plt.scatter(fitted_object.cluster_centers_[:, 0], fitted_object.cluster_centers_[:, 1], s = 100, c = 'k', label = 'Centroids', alpha = 0.5)
            plt.title('Clusters')
            plt.legend()
            plt.savefig('./kmeans/MallCustomers/test.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()

            plt.scatter(self.X[:,0], self.X[:,1], edgecolors = 'k')
            plt.title('Data Points')
            plt.grid(False)
            plt.savefig('./kmeans/MallCustomers/train.svg', format='svg', dpi=1200)
            # plt.show()
            plt.close()

            with open("./kmeans/MallCustomers/wcss.txt", "w") as text_file:
                print(round(fitted_object.inertia_, 2), file=text_file)

            with open("./kmeans/MallCustomers/time.txt", "w") as text_file:
                print(str(round(time.time()-self.start_time,2)) + " seconds", file=text_file)
        func, arg = {
            "polynomial_regression": (polynomialregression, ()),
            "linear_regression" : (linearregression, ()),
            "logistic_regression" : (logisticregression, ()),
            "knn" : (knn, ()),
            "svc" : (svc, ()),
            "decision_tree_classification" : (decisiontreeclassification, ()),
            "random_forest_classification" : (randomforestclassification, ()),
            "kmeans" : (kmeans, ()),
        }.get(algorithm, "failed to find the output data of chosen algorithm")
        return func(*arg)

class MLAlgorithm:
    def linearregression(self, X_train, y_train):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        return regressor
    def polynomialregression(self, X, y, pdegree = int(2)):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly_reg = PolynomialFeatures(degree = pdegree)
        X_poly = poly_reg.fit_transform(X)
        poly_reg.fit(X_poly, y)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        return (lin_reg, poly_reg)
    def logisticregression(self, X_train, y_train):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        return classifier
    def knn(self, X_train, y_train, n_neighbours = 5):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = n_neighbours)
        classifier.fit(X_train, y_train)
        return classifier
    def svc(self, X_train, y_train, kernel = 'rbf', C=1):
        from sklearn.svm import SVC
        classifier = SVC(kernel = kernel, random_state = 0, C=C)
        classifier.fit(X_train, y_train)
        return classifier
    def decisiontreeclassification(self, X_train, y_train, criterion = "gini", max_depth = None):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion, max_depth = max_depth , random_state = 0)
        classifier.fit(X_train, y_train)
        return classifier
    def randomforestclassification(self, X_train, y_train, n_estimators = 10, criterion = "gini", max_depth = None):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(criterion, n_estimators = n_estimators, max_depth = max_depth , random_state = 0)
        classifier.fit(X_train, y_train)
        return classifier
    def kmeans(self, X, n_clusters, init, n_init):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters = n_clusters, init = init, n_init = n_init)
        wcss = []
        for i in range(1, 11):
            temp_kmeans = KMeans(n_clusters = i, init = init, n_init = n_init)
            temp_kmeans.fit(X)
            wcss.append(temp_kmeans.inertia_)
        plt.plot(range(1, 11), wcss, color='red')
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('./kmeans/MallCustomers/kvswcss.svg', format='svg', dpi=1200)
        # plt.show()
        plt.close()
        return kmeans

Configuration.algorithm = form.getvalue('algo')
Configuration.data = form.getvalue('dataset')

Configuration.topic = "Regression"
if (Configuration.algorithm == "linear_regression"):
    d = Data()
    d.input(Configuration.algorithm)
    d.preprocess(Configuration.algorithm)
    algo = MLAlgorithm()
    object = algo.logisticregression(d.X_train, d.y_train)
    d.output(Configuration.algorithm, object)
elif (Configuration.algorithm == "logistic_regression"):
    t_size = int(form.getvalue('t_size'))
    d = Data()
    d.input(Configuration.algorithm, t_size = t_size)
    d.preprocess(Configuration.algorithm)
    algo = MLAlgorithm()
    object = algo.logisticregression(d.X_train, d.y_train)
    d.output(Configuration.algorithm, object)
elif (Configuration.algorithm == "knn"):
    t_size = int(form.getvalue('t_size'))
    n = int(form.getvalue('n_neighbours'))
    # kernels = ["rbf", "linear", "sigmoid", "poly"]
    d = Data()
    d.input(Configuration.algorithm, t_size = t_size)
    d.preprocess(Configuration.algorithm)
    algo = MLAlgorithm()
    object = algo.knn(d.X_train, d.y_train, n)
    d.output(Configuration.algorithm, object)
elif (Configuration.algorithm == "svc"):
    t_size = int(form.getvalue('t_size'))
    kernel = (form.getvalue('kernel'))
    C = float(form.getvalue('C'))
    # kernels = ["rbf", "linear", "sigmoid", "poly"]
    d = Data()
    d.input(Configuration.algorithm, t_size = t_size)
    d.preprocess(Configuration.algorithm)
    algo = MLAlgorithm()
    object = algo.svc(d.X_train, d.y_train, kernel, C)
    d.output(Configuration.algorithm, object, kernel)
elif (Configuration.algorithm == "decision_tree_classification"):
    t_size = int(form.getvalue('t_size'))
    max_depth = int(form.getvalue('max_depth'))
    criterion = form.getvalue('criterion')
    d = Data()
    d.input(Configuration.algorithm, t_size = t_size)
    d.preprocess(Configuration.algorithm)
    algo = MLAlgorithm()
    object = algo.decisiontreeclassification(d.X_train, d.y_train, criterion=criterion, max_depth = max_depth)
    d.output(Configuration.algorithm, object)
elif (Configuration.algorithm == "random_forest_classification"):
    t_size = int(form.getvalue('t_size'))
    max_depth = int(form.getvalue('max_depth'))
    criterion = form.getvalue('criterion')
    n_estimators = int(form.getvalue('n_estimators'))
    d = Data()
    d.input(Configuration.algorithm, t_size = t_size)
    d.preprocess(Configuration.algorithm)
    algo = MLAlgorithm()
    object = algo.decisiontreeclassification(d.X_train, d.y_train, criterion=criterion, max_depth = max_depth)
    d.output(Configuration.algorithm, object)
elif (Configuration.algorithm == "kmeans"):
    Configuration.data = "Mall_Customers.csv"
    n_clusters = int(form.getvalue('n_clusters'))
    init = form.getvalue('init')
    n_init = int(form.getvalue('n_init'))
    d = Data()
    d.input(Configuration.algorithm)
    d.preprocess(Configuration.algorithm)
    algo = MLAlgorithm()
    object = algo.kmeans(d.X, n_clusters=n_clusters, init = init, n_init = n_init)
    d.output(Configuration.algorithm, object)
