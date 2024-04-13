from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hex2color
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle
import warnings
import matplotlib
warnings.filterwarnings("ignore")


matplotlib.use('TkAgg')  # Set the backend
# Convert hex color to RGB
color_hex = "#fe595dff"
color_rgb = hex2color(color_hex)
# Create a custom colormap
cmap = LinearSegmentedColormap.from_list("mycmap", [(1, 1, 1), color_rgb], N=256)

class Classifier:
    def __init__(self, params_lr=None, params_svm=None, params_lda=None, params_qda=None,params_rfc=None):
        self.model_lr = LogisticRegression(**(params_lr if params_lr is not None else {}), n_jobs=-1)
        self.model_svm = SVC(**(params_svm if params_svm is not None else {}))
        self.model_lda = LinearDiscriminantAnalysis(**(params_lda if params_lda is not None else {}))
        self.model_qda = QuadraticDiscriminantAnalysis(**(params_qda if params_qda is not None else {}))
        self.model_tree = tree.DecisionTreeClassifier()
        self.model_rfc = RandomForestClassifier(**(params_rfc if params_rfc is not None else {}))

    def fit(self, X_train, y_train, X_val,model_name):
        if model_name == 'lr':
            self.model_lr.fit(X_train, y_train)
            y_pred = self.model_lr.predict(X_val)
            y_prob = self.model_lr.predict_proba(X_val)
        elif model_name == 'svm':
            self.model_svm.fit(X_train, y_train)
            y_pred = self.model_svm.predict(X_val)
            y_prob = self.model_svm.predict_proba(X_val)
        elif model_name == 'lda':
            self.model_lda.fit(X_train, y_train)
            y_pred = self.model_lda.predict(X_val)
            y_prob = self.model_lda.predict_proba(X_val)
        elif model_name == 'qda':
            self.model_qda.fit(X_train, y_train)
            y_pred = self.model_qda.predict(X_train)
            y_prob = self.model_qda.predict_proba(X_val)
        elif model_name == 'tree':
            self.model_tree.fit(X_train, y_train)
            y_pred = self.model_tree.predict(X_val)
            y_prob = self.model_tree.predict_proba(X_val)
        elif model_name == 'rfc':
            self.model_rfc.fit(X_train, y_train)
            y_pred = self.model_rfc.predict(X_val)
            y_prob = self.model_rfc.predict_proba(X_val)
        else:
            raise ValueError("Invalid model name. Please choose from 'lr', 'svm', 'lda', or 'qda'.")

        return y_pred, y_prob

    def tune(self, X_train, y_train, X_val, param_grid, model_name):
        if model_name == 'lr':
            grid = GridSearchCV(self.model_lr, param_grid, n_jobs=-1)
        elif model_name == 'svm':
            grid = GridSearchCV(self.model_svm, param_grid, n_jobs=-1)
        elif model_name == 'lda':
            grid = GridSearchCV(self.model_lda, param_grid, n_jobs=-1)
        elif model_name == 'qda':
            grid = GridSearchCV(self.model_qda, param_grid, n_jobs=-1)
        elif model_name == 'rfc':
            grid = GridSearchCV(self.model_rfc, param_grid, n_jobs=-1)
        else:
            raise ValueError("Invalid model name. Please choose from 'lr', 'svm', 'lda', or 'qda'.")

        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        y_pred_opt = grid.predict(X_val)
        return best_params, y_pred_opt

    
    
    def report(self, y_pred_opt, y_val, y_prob):
        auc_score = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')
        accuracy = accuracy_score(y_val, y_pred_opt)
        report = classification_report(y_val, y_pred_opt)
        confusion = confusion_matrix(y_val, y_pred_opt)
        percision = precision_score(y_val, y_pred_opt, average='weighted')
        recall = recall_score(y_val, y_pred_opt, average='weighted')
        f1 = f1_score(y_val, y_pred_opt, average='weighted')
        support = np.unique(y_val, return_counts=True)[1]
        return auc_score, accuracy, report, confusion, percision, recall, f1, support


    def plot(self, confusion, y_prob, y_val,  save_fig=False, filename='model_plot.png'):
        # Initialize the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

        # Confusion Matrix
        sns.heatmap(confusion, annot=True, fmt='d', cmap=cmap, ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        # ROC Curve
        n_classes = y_prob.shape[1]
        y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'black'])
        for i, color in zip(range(n_classes), colors):
            ax2.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0.0, 1.0])
        ## After see all roc curve, set the ylim start from 0.6
        ax2.set_ylim([0.0, 1.0])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc='lower right')

        plt.tight_layout()
        if save_fig:
            plt.savefig(filename, format='png', dpi=300)  
        plt.show()


## Data Processing，we decided to use features from Chi-square test as the input for the model
df1 = pd.read_csv('data/cleaned.csv')

X = df1[['host_response_time', 'host_response_rate', 'host_is_superhost',
       'room_type', 'accommodates', 'minimum_nights', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_value', 'instant_bookable', 'reviews_per_month',
       'host_years', 'review_years_range']]
y = df1['popularity']

## Normalize X
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Define the parameters for the models
params_lr={'max_iter': 1000, 'C': 0.5}
params_svm={'kernel': 'linear', 'C': 0.5, 'probability': True}
params_lda = {'solver': 'svd'}
params_qda = {'reg_param': 0.5}
params_rfc = {'max_depth': 5}

## Define the parameter grid for the models
params_lr_grid = {'max_iter': [1000, 2000, 3000], 'C': [0.1, 0.5, 1.0],
                  'penalty': ['l1', 'l2','none'],
                  'solver': ['liblinear', 'saga']}

params_svm_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                   'C': [0.1, 0.5, 1.0],
                   'gamma': ['scale', 'auto']}

params_lda_grid = {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]}
params_qda_grid = {'reg_param': [0.1, 1, 10]}
params_rfc_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}


# Logistic Regression
model_lrg = Classifier(params_lr=params_lr)

## Fit the model
y_pred_lrg, y_prob = model_lrg.fit(X_train, y_train, X_test,'lr')

## Fine tune the model and get the best parameters
best_parameters, y_pred_opt = model_lrg.tune(X_train, y_train, X_test, params_lr_grid, 'lr')
print(f"Best Parameters: {best_parameters}")

## Get the report for each metric
auc_score,lrg_accuracy, report, confusion, percision, recall, f1, support = model_lrg.report(y_pred_opt, y_test, y_prob)

print("\nLogistic Regression")
print(f"AUC: {auc_score}")
print(f"Accuracy: {lrg_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

## Plot the confusion matrix and ROC curve
model_lrg.plot(confusion, y_prob, y_test,save_fig=True, filename='logistic_regression_analysis.png')



# Support Vector Machine
model_svm = Classifier(params_svm=params_svm)

## Fit the model
y_pred_svm, y_prob = model_svm.fit(X_train, y_train, X_test,'svm')

## Fine tune the model and get the best parameters
best_parameters, y_pred_opt = model_svm.tune(X_train, y_train, X_test, params_svm_grid, 'svm')
print(f"Best Parameters: {best_parameters}")

## Get the report for each metric
auc_score, svm_accuracy, report, confusion, percision, recall, f1, support = model_svm.report(y_pred_opt, y_test, y_prob)

print("\n Support Vector Machine")
print(f"AUC: {auc_score}")
print(f"Accuracy: {svm_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

## Plot the confusion matrix and ROC curve
model_svm.plot(confusion, y_prob, y_test, save_fig=True, filename='support_vector_machine_analysis.png')



# Linear Discriminant Analysis
model_lda = Classifier(params_lda=params_lda)

## Fit the model
y_pred_lda, y_prob = model_lda.fit(X_train, y_train, X_test,'lda')

## Fine tune the model and get the best parameters
best_parameters, y_pred_opt = model_lda.tune(X_train, y_train, X_test, params_lda_grid, 'lda')
print(f"Best Parameters: {best_parameters}")

## Get the report for each metric
auc_score, lda_accuracy, report, confusion, percision, recall, f1, support = model_lda.report(y_pred_opt, y_test, y_prob)

print("\nLinear Discriminant Analysis")
print(f"AUC: {auc_score}")
print(f"Accuracy: {lda_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

## Plot the confusion matrix and ROC curve
model_lda.plot(confusion, y_prob, y_test, save_fig=True, filename='linear_discriminant_analysis.png')




# Quadratic Discriminant Analysis
model_qda = Classifier(params_qda=params_qda)

## Fit the model
y_pred_qda, y_prob = model_qda.fit(X_train, y_train, X_test,'qda')

## Fine tune the model and get the best parameters
best_parameters, y_pred_opt = model_qda.tune(X_train, y_train, X_test, params_qda_grid, 'qda')
print(f"Best Parameters: {best_parameters}")

## Get the report for each metric
auc_score, qda_accuracy, report, confusion, percision, recall, f1, support = model_qda.report(y_pred_opt, y_test, y_prob)

print("\nQuadratic Discriminant Analysis")
print(f"AUC: {auc_score}")
print(f"Accuracy: {qda_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

## Plot the confusion matrix and ROC curve
model_qda.plot(confusion, y_prob, y_test, save_fig=True, filename='quadratic_discriminant_analysis.png')




# Random Forest Classifier
model_rfc = Classifier(params_rfc=params_rfc)

y_pred_rfc, y_prob_rfc = model_rfc.fit(X_train, y_train, X_test, 'rfc')


best_parameters, y_pred_opt = model_rfc.tune(X_train, y_train, X_test, params_rfc_grid, 'rfc')
print(f"Best parameters: {best_parameters}")

auc_score, rfc_accuracy, report, confusion, percision, recall, f1, support = model_rfc.report(y_pred_opt, y_test, y_prob_rfc)

print(f"AUC Score: {auc_score}")
print(f"Accuracy: {rfc_accuracy}")
print(f"Report: {report}")
print(f"Precision: {percision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Support: {support}")

model_rfc.plot(confusion, y_prob_rfc, y_test, save_fig=True, filename='random_forest_classifier.png')



# Decision Tree Classifier
model_tree = Classifier()
y_pred_rfc, y_prob_rfc  = model_tree.fit(X_train, y_train, X_test, 'tree')

auc_score, tree_accuracy, report, confusion, percision, recall, f1, support = model_tree.report(y_pred_rfc, y_test, y_prob_rfc)

print(f"AUC Score: {auc_score}")
print(f"Accuracy: {tree_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

model_tree.plot(confusion, y_prob_rfc, y_test, save_fig=True, filename='decision_tree_classifier.png')




## Model Comparison
table = pd.DataFrame({'Model': ['LR', 'SVM', 'LDA', 'QDA', 'RFC', 'Tree'],
                      'Accuracy': [lrg_accuracy, svm_accuracy, lda_accuracy, qda_accuracy, rfc_accuracy, tree_accuracy]})

color_palette = sns.color_palette("Set2")  # Using seaborn to get a set of colors
color_dict = dict(zip(table['Model'], color_palette))
marker_dict = {'LR': 'o', 'SVM': 's', 'LDA': '^', 'QDA': 'p', 'RFC': '*', 'Tree': 'D'}

# Get current axis
def plot_model_comparison(save_fig=False, filename='model_comparison.png'):
    ax = plt.gca()
    ax.set_facecolor('#f8f8f8')  # A slightly darker shade of grey for the face

    # Grid lines can be softer and less pronounced
    ax.grid(True, which='both', axis='both', linestyle='-', linewidth=0.75, color='lightgrey', alpha=0.5)

    # Create a scatter plot with different markers and colors
    for model, row in table.iterrows():
        ax.scatter(row['Model'], row['Accuracy'], color=color_dict[row['Model']],
                marker=marker_dict[row['Model']], label=row['Model'],
                s=150, edgecolors='black', linewidths=0.5)
        
    # Adding labels and title with increased font size
    plt.title('Model Accuracy', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(np.arange(0.4, 1.1, 0.1), fontsize=12)

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Leave only bottom and left spines and make them lighter and less pronounced
    ax.spines['left'].set_color('lightgrey')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['bottom'].set_linewidth(0.5)

    # Adding a legend outside of the plot with optimized layout and font size
    leg = ax.legend(title="Model", loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=12)
    plt.setp(leg.get_title(), fontsize=14)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename, format='png', dpi=300)
        print(f"Saved figure as {filename}")
    plt.show()

plot_model_comparison(save_fig=True, filename='model_comparison.png')