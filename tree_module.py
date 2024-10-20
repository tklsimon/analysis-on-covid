import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def print_feature_importance(fitted_model) -> None:
    # Random Forest feature importance
    feat_name_list = fitted_model.feature_names_in_
    impt_list = fitted_model.feature_importances_
    impt_order_index_list = np.argsort(impt_list)[::-1]
    
    # Print feature importance
    print("Feature Importances:")
    for i in range(len(feat_name_list)):
        print(f'{i+1}. {feat_name_list[impt_order_index_list[i]]}: {impt_list[impt_order_index_list[i]]:.4f}')


def print_random_forest_importance(X, y, random_state) -> None:
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # # Regression Tree
    # reg_tree_model = DecisionTreeRegressor(random_state=random_state)
    # reg_tree_model.fit(X_train, y_train)
    # reg_tree_y_pred = reg_tree_model.predict(X_test)
    # print(f'Decision Tree MSE: {mean_squared_error(reg_tree_y_pred, y_test): .6}')
    # print_feature_importance(reg_tree_model)
    
    # Random Forest
    random_forest_model = RandomForestRegressor(random_state=random_state)
    random_forest_model.fit(X_train, y_train)
    random_forest_y_pred = random_forest_model.predict(X_test)
    print('Random Forest')
    print(f'MSE: {mean_squared_error(random_forest_y_pred, y_test): .6}')
    print_feature_importance(random_forest_model)
