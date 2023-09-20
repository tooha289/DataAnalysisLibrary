import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class DataAnalysisVisualizer:
    """
    A class providing visualization functions for data analysis using Seaborn and Matplotlib.

    This class includes three commonly used data visualization functions:
    - draw_scatterplot: Create scatter plots for multiple x_columns against a single y_column.
    - draw_histplot: Create histogram plots for specified columns.
    - draw_countplot: Create count plots for specified columns.

    Parameters:
    - None"""

    def draw_scatterplot(
        self,
        data,
        x_columns,
        y_column,
        hue_column=None,
        figsize=(8, 6),
        figshape=None,
        rotate_xlabel=False,
    ):
        """
        Create scatter plots for multiple x_columns against a single y_column using Seaborn.

        Parameters:
        - data (DataFrame): The DataFrame containing the data.
        - x_columns (list of str): List of column names to be used as the x-axis variables.
        - y_column (str): The column name to be used as the y-axis variable.
        - hue_column (str, optional): The column name to be used for coloring the points. Default is None.
        - figsize (tuple, optional): Figure size in inches (width, height). Default is (8, 6).
        - figshape (tuple, optional): Number of rows and columns for the subplot grid. Default is (len(x_columns), 1).
        - rotate_xlabel (bool, optional): Whether to rotate x-axis labels by 90 degrees. Default is False.


        Example:
        draw_scatterplot(dataframe, ['x1', 'x2', 'x3'], 'y', hue_column='hue_column', figsize=(12, 6), figshape=(2, 2))
        draw_scatterplot(dataframe, ['x1', 'x2', 'x3'], 'y', figsize=(12, 6), figshape=(2, 2))
        """
        # Use the default figsize if not specified
        if figsize is None:
            figsize = (8, 6)

        # Use the default figshape if not specified
        if figshape is None:
            figshape = (len(x_columns), 1)

        # Set the number of rows and columns for the subplots
        num_rows, num_cols = figshape

        # Calculate the total number of subplots required
        total_subplots = len(x_columns)

        # Create subplots for plotting
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

        # Flatten axes for sequential access
        axes = axes.flatten()

        # Create scatter plots for each x_column against y_column
        for i, x_column in enumerate(x_columns):
            # Check if the current subplot index is within the valid range of subplots
            if i >= total_subplots:
                break
            ax = axes[i]

            # Use hue_column for coloring points if specified
            if hue_column is not None:
                sns.scatterplot(
                    data=data,
                    x=x_column,
                    y=y_column,
                    hue=hue_column,
                    ax=ax,
                )
            else:
                sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax)

            # Set x-axis label with rotation if rotate_xlabel is True
            if rotate_xlabel:
                ax.set_xlabel(x_column, rotation=90)
            else:
                ax.set_xlabel(x_column)
            # Set y-axis labels
            ax.set_ylabel(y_column)

            ax.set_title(f"Scatter Plot for {x_column}")

        # Check if there are more available subplot axes than columns to plot
        if len(x_columns) < len(axes):
            # If there are unused axes, remove them from the figure
            unused_axes = axes[len(x_columns) :]
            for unused_axe in unused_axes:
                fig.delaxes(unused_axe)

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plots
        plt.show()

    def draw_histplot(
        self,
        data,
        columns,
        hue_column=None,
        figsize=None,
        figshape=None,
        kde=True,
        rotate_xlabel=False,
    ):
        """
        Create histogram plots for specified columns in a DataFrame using Seaborn.

        Parameters:
        - data (DataFrame): The DataFrame containing the data.
        - columns (list of str): List of column names to be used for the histograms.
        - hue_column (str, optional): The column name to be used for coloring the histograms. Default is None.
        - figsize (tuple, optional): Figure size in inches (width, height). Default is None.
        - figshape (tuple, optional): Number of rows and columns for the subplot grid. Default is None.
        - kde (bool, optional): Whether to overlay Kernel Density Estimation (KDE) on the histograms. Default is True.
        - rotate_xlabel (bool, optional): Whether to rotate x-axis labels by 90 degrees. Default is False.


        Example:
        draw_histplot(dataframe, ['column1', 'column2', 'column3'], hue_column='hue_column', figsize=(12, 6), figshape=(2, 2))
        draw_histplot(dataframe, ['column1', 'column2', 'column3'], figsize=(12, 6), figshape=(2, 2))
        """
        # Use the default figsize if not specified
        if figsize is None:
            figsize = (8, 6)

        # Use the default figshape if not specified
        if figshape is None:
            figshape = (len(columns), 1)

        # Set the number of rows and columns for the subplots
        num_rows, num_cols = figshape

        # Calculate the total number of subplots required
        total_subplots = len(columns)

        # Create subplots for plotting
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

        # Flatten axes for sequential access
        axes = axes.flatten()

        # Create histograms for each specified column
        for i, column in enumerate(columns):
            # Check if the current subplot index is within the valid range of subplots
            if i >= total_subplots:
                break
            ax = axes[i]

            # Use hue_column for coloring histograms if specified
            if hue_column is not None:
                sns.histplot(
                    data=data,
                    x=column,
                    hue=hue_column,
                    multiple="stack",
                    ax=ax,
                    kde=kde,
                )
            else:
                sns.histplot(data=data, x=column, ax=ax, kde=kde)

            # Set x-axis label with rotation if rotate_xlabel is True
            if rotate_xlabel:
                ax.set_xlabel(column, rotation=90)
            else:
                ax.set_xlabel(column)

            # Set title for the histogram
            ax.set_title(f"Histogram for {column}")

        # Check if there are more available subplot axes than columns to plot
        if len(columns) < len(axes):
            # If there are unused axes, remove them from the figure
            unused_axes = axes[len(columns) :]
            for unused_axe in unused_axes:
                fig.delaxes(unused_axe)

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plots
        plt.show()

    def draw_countplot(
        self,
        data,
        columns,
        hue_column=None,
        figsize=None,
        figshape=None,
        annotation=True,
        rotate_xlabel=False,
    ):
        """
        Create count plots for specified columns in a DataFrame using Seaborn.

        Parameters:
        - data (DataFrame): The DataFrame containing the data.
        - columns (list of str): List of column names to be used for the count plots.
        - hue_column (str, optional): The column name to be used for coloring the count plots.
        Should be of type str or categorical. Default is None.
        - figsize (tuple, optional): Figure size in inches (width, height). Default is None.
        - figshape (tuple, optional): Number of rows and columns for the subplot grid. Default is None.
        - annotation (bool, optional): Whether to annotate each bar with its count value. Default is True.
        - rotate_xlabel (bool, optional): Whether to rotate x-axis labels by 90 degrees. Default is False.


        Example:
        draw_countplot(dataframe, ['column1', 'column2', 'column3'], hue_column='hue_column', figsize=(12, 6), figshape=(2, 2), annotation=False)
        draw_countplot(dataframe, ['column1', 'column2', 'column3'], figsize=(12, 6), figshape=(2, 2))
        """
        # Use the default figsize if not specified
        if figsize is None:
            figsize = (8, 6)

        # Use the default figshape if not specified
        if figshape is None:
            figshape = (len(columns), 1)

        # Set the number of rows and columns for the subplots
        num_rows, num_cols = figshape

        # Calculate the total number of subplots required
        total_subplots = len(columns)

        # Create subplots for plotting
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

        # Flatten axes for sequential access
        axes = axes.flatten()

        # Create count plots for each specified column
        for i, column in enumerate(columns):
            # Check if the current subplot index is within the valid range of subplots
            if i >= total_subplots:
                break
            ax = axes[i]

            # Use hue_column for coloring count plots if specified
            if hue_column is not None:
                sns.countplot(data=data, x=column, hue=hue_column, ax=ax)
            else:
                sns.countplot(data=data, x=column, ax=ax)

            # Set x-axis label with rotation if rotate_xlabel is True
            if rotate_xlabel:
                ax.set_xlabel(columns, rotation=90)
            else:
                ax.set_xlabel(columns)

            # Set title for the count plot
            ax.set_title(f"Count Plot for {column}")

            # Annotate each bar with its count value
            if annotation == True:
                for p in ax.patches:
                    ax.annotate(
                        f"{p.get_height()}",
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center",
                        va="bottom",
                    )

        # Check if there are more available subplot axes than columns to plot
        if len(columns) < len(axes):
            # If there are unused axes, remove them from the figure
            unused_axes = axes[len(columns) :]
            for unused_axe in unused_axes:
                fig.delaxes(unused_axe)

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plots
        plt.show()
