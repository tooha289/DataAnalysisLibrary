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
        rotate_xticks=False,
        **params,
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
        - rotate_xticks (bool, optional): Whether to rotate x-axis ticks labels by 90 degrees. Default is False.
        - **params: Additional keyword arguments to be passed to the Seaborn pairplot function.

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
        axes = np.array(axes).flatten()

        # Create scatter plots for each x_column against y_column
        for i, x_column in enumerate(x_columns):
            # Check if the current subplot index is within the valid range of subplots
            if i >= total_subplots:
                break
            ax = axes[i]

            # Use hue_column for coloring points if specified
            if hue_column is not None:
                sns.scatterplot(
                    data=data, x=x_column, y=y_column, hue=hue_column, ax=ax, **params
                )
            else:
                sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax, **params)

            # Set x-axis labels and y-axis labels
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

            # Set x-axis ticks label rotation if rotate_xticks is True
            if rotate_xticks:
                ax.tick_params(axis="x", labelrotation=90)

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
        rotate_xticks=False,
        **params,
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
        - rotate_xticks (bool, optional): Whether to rotate x-axis ticks labels by 90 degrees. Default is False.
        - **params: Additional keyword arguments to be passed to the Seaborn pairplot function.

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
        axes = np.array(axes).flatten()

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
                    **params,
                )
            else:
                sns.histplot(data=data, x=column, ax=ax, kde=kde, **params)

            # Set x-axis ticks label rotation if rotate_xticks is True
            if rotate_xticks:
                ax.tick_params(axis="x", labelrotation=90)

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
        rotate_xticks=False,
        **params,
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
        - rotate_xticks (bool, optional): Whether to rotate x-axis ticks labels by 90 degrees. Default is False.
        - **params: Additional keyword arguments to be passed to the Seaborn pairplot function.

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
        axes = np.array(axes).flatten()

        # Create count plots for each specified column
        for i, column in enumerate(columns):
            # Check if the current subplot index is within the valid range of subplots
            if i >= total_subplots:
                break
            ax = axes[i]

            # Use hue_column for coloring count plots if specified
            if hue_column is not None:
                sns.countplot(data=data, x=column, hue=hue_column, ax=ax, **params)
            else:
                sns.countplot(data=data, x=column, ax=ax, **params)

            # Set x-axis ticks label rotation if rotate_xticks is True
            if rotate_xticks:
                ax.tick_params(axis="x", labelrotation=90)

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

    def draw_pairplot(
        self,
        data,
        columns,
        hue_column=None,
        rotate_xticks=False,
        diag_kind="auto",
        **params,
    ):
        """
        Create a pairplot for specified columns in a DataFrame using Seaborn.

        Parameters:
        - data (DataFrame): The DataFrame containing the data.
        - columns (list of str): List of column names to be used for the pairplot.
        - hue_column (str, optional): The column name to be used for coloring the pairplot.
        Should be of type str or categorical. Default is None.
        - rotate_xticks (bool, optional): Whether to rotate x-axis ticks labels by 90 degrees. Default is False.
        - diag_kind (str, optional): The type of plot to use for the diagonal subplots. Options are "auto" (default), "hist", "kde", or "scatter".
        - **params: Additional keyword arguments to be passed to the Seaborn pairplot function.

        Example:
        draw_pairplot(dataframe, ['column1', 'column2', 'column3'], hue_column='hue_column')
        draw_pairplot(dataframe, ['column1', 'column2', 'column3'])
        """
        # Create pairplot
        if hue_column is not None:
            sns.set(style="ticks")
            grid = sns.pairplot(
                data=data, vars=columns, hue=hue_column, diag_kind=diag_kind, **params
            )
        else:
            sns.set(style="ticks")
            grid = sns.pairplot(data=data, vars=columns, diag_kind=diag_kind, **params)

        # Set x-axis ticks label rotation if rotate_xticks is True
        if rotate_xticks:
            for ax in grid.axes.flat:
                ax.tick_params(axis="x", labelrotation=90)

        # Display the pairplot
        plt.show()

    def draw_heatmap(
        self,
        corr_matrix,
        figsize=None,
        annotation_format=".1f",
        **params,
    ):
        """
        Create a heatmap for a correlation matrix using Seaborn.

        Parameters:
        - corr_matrix (DataFrame): The correlation matrix to be visualized.
        - figsize (tuple, optional): Figure size in inches (width, height). Default is None.
        - annotation_format (str, optional): Format string for annotating cells in the heatmap. Default is ".1f" (one decimal place).
        - **params: Additional keyword arguments to be passed to the Seaborn heatmap function.

        Example:
        draw_heatmap(correlation_matrix)
        draw_heatmap(correlation_matrix, figsize=(8, 6), cmap="coolwarm", annot=True, annotation_format=".2f")
        """
        # Use the default figsize if not specified
        if figsize is None:
            figsize = (8, 6)

        # Create the heatmap
        plt.figure(figsize=figsize)
        sns.set(font_scale=1)  # Adjust font size if needed
        sns.heatmap(
            data=corr_matrix,
            annot=True,
            fmt=annotation_format,
            **params,
        )

        # Display the heatmap
        plt.show()

    def draw_boxplot(
        self, data, columns, x_axis=True, figsize=None, figshape=None, **params
    ):
        """
        Create box plots for specified columns in a DataFrame using Seaborn.

        Parameters:
        - data (DataFrame): The DataFrame containing the data.
        - columns (list of str): List of column names to be used for the box plots.
        - x_axis (bool, optional): Whether to use x-axis (True) or y-axis (False) for the box plots. Default is True.
        - figsize (tuple, optional): Figure size in inches (width, height). Default is None.
        - figshape (tuple, optional): Number of rows and columns for the subplot grid. Default is None.
        - **params: Additional keyword arguments to be passed to the Seaborn boxplot function.

        Example:
        draw_boxplot(dataframe, ['column1', 'column2', 'column3'], figsize=(12, 6), figshape=(2, 2), notch=True)
        draw_boxplot(dataframe, ['column1', 'column2', 'column3'], figsize=(12, 6), notch=True)
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
        axes = np.array(axes).flatten()

        # Create box plots for each specified column
        for i, column in enumerate(columns):
            # Check if the current subplot index is within the valid range of subplots
            if i >= total_subplots:
                break
            ax = axes[i]

            # Determine whether to use x-axis or y-axis based on the x_axis parameter
            if x_axis:
                sns.boxplot(data=data, x=column, ax=ax, **params)
            else:
                sns.boxplot(data=data, y=column, ax=ax, **params)

            # Set title for the box plot
            ax.set_title(f"Box Plot for {column}")

        # Check if there are more available subplot axes than columns to plot
        if len(columns) < len(axes):
            # If there are unused axes, remove them from the figure
            unused_axes = axes[len(columns) :]
            for unused_ax in unused_axes:
                fig.delaxes(unused_ax)

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plots
        plt.show()
