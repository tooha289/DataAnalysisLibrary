import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import permutation_importance
import statsmodels.formula.api as smf


class DataFramePreprocessor(object):
    """This class helps perform preprocessing on data frames.
    Through this class, you can perform scaling or encoding operations on data frames.

    The currently verified transformers are as follows.
    MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def fit_transform_multiple_transformer(
        self, dataframe, transformers, columns_matrix, fit=True
    ):
        """Transforms each column of the DataFrame using copied transformers
        and returns the transformed DataFrame along with the fitted transformers.

        Args:
           df: The DataFrame to perform the transformation on.
           transformers: The source transformers on which to perform the transformation operation.
           The transformer in question must have a 'fit_transform' function.
           columns: This is a matrix of columns to be applied to each transformer.
           fit: Whether to perform fit before transform.

        Return:
            result_df: It is a DataFrame containing only the transformed data.
            transformers_dict: This is a dict with each column name as the key and the transformer as the value.
        """
        result_dfs = []
        result_tfs = []
        try:
            for transformer, columns in zip(transformers, columns_matrix):
                df, tfs = self.fit_transform_single_transformer(
                    dataframe, transformer, columns, fit
                )
                result_dfs.append(df)
                result_tfs.append(tfs)

            result_df = pd.concat(result_dfs, axis=1)
            transformers_dict = {k: v for dic in result_tfs for k, v in dic.items()}
            return (result_df, transformers_dict)
        except Exception as e:
            self._logger.exception(e)
            return None

    def fit_transform_single_transformer(
        self, dataframe, transformer, columns, fit=True
    ):
        """Transforms each column of the DataFrame using copied transformers
        and returns the transformed DataFrame along with the fitted transformers.

        Args:
           dataframe: The DataFrame to perform the transformation on.
           transformer: The source transformer on which to perform the transformation operation.
           The transformer in question must have a 'fit_transform' function.
           columns: List of columns on which transformations will be applied in the DataFrame.
           fit: Whether to perform fit before transform.

        Return:
            result_df: It is a DataFrame containing only the transformed data.
            transformers_dict: This is a dict with each column name as the key and the transformer as the value.
        """
        if len(columns) < 1:
            return None
        # The transformer corresponding to each column is stored.
        transformers = {}
        # Variable to store the transformed data
        data = []

        try:
            # Perform transformations on each column.
            for col in columns:
                tf = deepcopy(transformer)
                df_col = dataframe[col].to_numpy()

                if isinstance(tf, LabelEncoder) == False:
                    df_col = df_col.reshape(-1, 1)

                if fit == True:
                    result = tf.fit_transform(df_col)
                else:
                    result = tf.transform(df_col)

                if len(result.shape) == 1:
                    result = result.reshape(-1, 1)
                data.append(result)
                transformers[col] = tf

            if isinstance(tf, OneHotEncoder):
                dfs = []
                cols = [
                    [f"{col}_{cat}" for cat in tf.categories_[0]]
                    for col, tf in transformers.items()
                ]
                for i, d in enumerate(data):
                    df = pd.DataFrame(d.toarray(), columns=cols[i])
                    dfs.append(df)
                result_df = pd.concat(dfs, axis=1)
            else:
                result_df = pd.DataFrame(np.hstack(data), columns=columns)

            return (result_df, transformers)
        except Exception as e:
            self._logger.exception(e)
            return None

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return super.__repr__()


class FeatureSelector(object):
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_permutation_importance(self, estimator, X, y):
        """Returns a "permutation_importance" DataFrame.

        Args:
            estimator : object
            An estimator that has already been :term:`fitted` and is compatible
            with :term:`scorer`.

            X : ndarray or DataFrame, shape (n_samples, n_features)
                Data on which permutation importance will be computed.

            y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
                Targets for supervised or `None` for unsupervised.
        """
        perm_importance = permutation_importance(estimator, X, y)
        pi = pd.DataFrame()
        pi["feature"] = X.columns
        pi["perm_importance"] = perm_importance.importances_mean
        return pi

    def get_vif_dataframe(self, formula, dataframe):
        y, X = dmatrices(formula, dataframe, return_type="dataframe")

        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF Factor"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]
        vif = vif.sort_values("VIF Factor", ascending=False)
        return vif

    def vif_analysis(self, formula, dataframe):
        while True:
            model = smf.ols(formula=formula, data=dataframe).fit()
            print(model.summary())

            vif_df = self.get_vif_dataframe(formula, dataframe)
            print(vif_df.to_string(index=False))

            if len(vif_df) < 1:
                return
            feature, vif = vif_df.loc[0, :].values
            if vif < 10:
                print("The analysis ends because there are no VIF values exceeding 10.")
                return

            print(f"Remove the {feature} feature with a VIF greater than 10.")
            find_str = f"+{feature}"
            if feature == "Intercept":
                formula += "-1"
            if formula.find(find_str) != -1:
                formula.replace(find_str, "")
            elif formula.find(feature) != -1:
                formula.replace(feature, "")

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return super.__repr__()


if __name__ == "__main__":
    # DataFramePreprocessor example
    # titanic = pd.read_csv(r"D:\Python\AI_Example\data\titanic.csv")
    # lbl = LabelEncoder()
    # mms = MinMaxScaler()
    # ohe = OneHotEncoder()

    # numeric_cols = ["Parch", "Age", "Fare"]
    # onehot_col = ["SibSp", "Sex"]
    # label_col = ["Pclass"]

    # dfp = DataFramePreprocessor()
    # df, tfs = dfp.fit_transform_single_transformer(titanic, lbl, onehot_col)
    # df, tfs = dfp.fit_transform_multiple_transformer(
    #     titanic, [mms, ohe, lbl], [numeric_cols, onehot_col, label_col]
    # )

    # col = [[k] for k in tfs]
    # transforms = [tf for tf in tfs.values()]
    # df, tfs = dfp.fit_transform_multiple_transformer(
    #     titanic[5:10], transforms, col, fit=False
    # )

    # FeatureSelector Example
    titanic = pd.read_csv("data/titanic_train.csv")
    titanic.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    lbe = LabelEncoder()
    dfp = DataFramePreprocessor()
    df, tfs = dfp.fit_transform_single_transformer(titanic, lbe, ["Sex", "Embarked"])

    titanic.update(df)
    titanic[["Sex", "Embarked"]] = titanic[["Sex", "Embarked"]].astype("int32")

    formula = "Survived~" + "+".join(titanic.columns.difference(["Survived"]))
    fs = FeatureSelector()
    fs.vif_analysis(formula, titanic)
