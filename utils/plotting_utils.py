import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_avg_metrics_by_level(df, metrics, labels, colors=None, model_order=None):

    df_melted = df.melt(id_vars=['model', 'level'], 
                        value_vars=metrics, 
                        var_name='metric', 
                        value_name='score')

    # List of levels
    levels = [1,2,3]

    # Calculate global min and max for y-axis
    ymin = 0
    ymax = max(1, df_melted.score.max())

    # Define the desired width space
    width_space = 0.1  # Adjust this value to control the gap between the level bar plots

    # Create subplots
    fig, axes = plt.subplots(1, len(levels), figsize=(12, 6),
                            gridspec_kw={'wspace': width_space}) # Increased figsize

    # Loop through levels and create a barplot for each
    for i, level in enumerate(levels):
        df_level = df_melted[df_melted['level'] == level]

        if i == 2:  
            ax = sns.barplot(
                x='metric',
                y='score', 
                hue='model', 
                data=df_level, 
                ax=axes[i], 
                legend=True, 
                palette=colors,
                linewidth=0.5,
                edgecolor="black",
                hue_order=model_order)
            # Move legend to top-left corner and make it larger
            ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.1, 1))  
        else:
            ax = sns.barplot(
                x='metric', 
                y='score', 
                hue='model', 
                data=df_level, 
                ax=axes[i], 
                legend=False, 
                palette=colors,
                linewidth=0.5,
                 edgecolor="black",
                hue_order=model_order)

        ax.set_xlabel(f'Level {level}')
        ax.set_xticklabels(labels)
        if i ==0:
            ax.set_ylabel('Score')
        else:
            ax.set_ylabel('')
        #ax.set_facecolor('black')
        ax.tick_params(axis='x', rotation=45, labelsize=8)  # Reduced labelsize
        ax.tick_params(axis='y', labelsize=8)  # Reduced labelsize
        #ax.spines['bottom'].set_color('white')
        #ax.spines['top'].set_color('white')
        ax.spines['left'].set_visible(False) # Remove left spine
        ax.spines['right'].set_visible(False)
        if i > 0: #Remove y axis for all plots other than the first
            ax.set_yticks([])
        # Set the y-axis limits for all subplots to be the same
        ax.set_ylim(ymin, ymax)

    # Add a horizontal line across the top and bottom of all subplots
    fig.subplots_adjust(top=0.9) # Adjust spacing at the top if necessary

    plt.tight_layout(rect=[0, -0.03 , .95])
    return fig, axes



def plot_perclass_metrics(df, metric, y_label, models, colors, ordered_labels=None):
    # Prepare the data for plotting
    grouped_data = df.groupby(['label', 'model'])[[metric]].mean().reset_index()

    # Pivot the data for recall and precision
    data = grouped_data.pivot(index='label', columns='model', values=metric)

    # Reorder the data based on the specified order
    if ordered_labels is not None:
        data = data.reindex(ordered_labels)

    # Plotting parameters
    labels = data.index  # Labels on the x-axis
    x = np.arange(len(labels)) * 1.3  # Increase space between each group of bars
    bar_width = 0.2  # Width of each bar

    # Create a figure and axis with a black background
    figsize = (15, 8)
    fig, ax = plt.subplots(figsize=figsize)
    #ax.set_facecolor('black')  # Set the background color of the plot area

    # Plot recall bars (above x-axis)
    for i, model in enumerate(models):
        ax.bar(
            x + i * bar_width,
            data[model],
            width=bar_width,
            color=colors[model],
            linewidth=0.5,
            edgecolor="black"
        )

    # Add a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8)

    # Customize y-axis ticks to show absolute values
    if data.min().min() < 0:
        ymin = max(-1, data.min().min() - 0.1)
    else:
        ymin = 0
    ymax = min(data.max().max() + 0.1, 1)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, num=10))
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(ymin, ymax, num=10)])
    ax.set_ylabel(y_label)


    # Customize the legend to show each model only once and make the title white
    custom_legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[model], label=model, edgecolor="black", linewidth=0.5) for model in models
    ]
    legend = ax.legend(handles=custom_legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.setp(legend.get_texts())  # Set legend text color to white

    ax.set_xticks(x + (len(models) - 1) * bar_width / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Adjust layout for better visibility
    plt.tight_layout()
    return fig, ax



def plot_perclas_double_bar(df, metrics, models, colors, ordered_labels):
    metric1_name, metric2_name = metrics.keys()
    metric1, metric2 = metrics.values()
    # Prepare the data for plotting
    grouped_data = df.groupby(['label', 'model'])[[metric1, metric2]].mean().reset_index()

    # Pivot the data for recall and precision
    metric1_data = grouped_data.pivot(index='label', columns='model', values=metric1)
    metric2_data = grouped_data.pivot(index='label', columns='model', values=metric2)

    # Convert precision to negative values for plotting below the x-axis
    metric2_data = -metric2_data

    # Reorder the data based on the specified order
    metric1_data = metric1_data.reindex(ordered_labels)
    metric2_data = metric2_data.reindex(ordered_labels)

    # Plotting parameters
    labels = metric1_data.index  # Labels on the x-axis
    x = np.arange(len(labels)) * 1.3  # X positions of labels
    bar_width = 0.2  # Width of each bar

    # Create a figure and axis with a black background
    figsize = (10,6)
    fig, ax = plt.subplots(figsize=figsize)
    #ax.set_facecolor('black')  # Set the background color of the plot area

    # Plot recall bars (above x-axis)
    for i, model in enumerate(models):
        ax.bar(
            x + i * bar_width,
            metric1_data[model],
            width=bar_width,
            color=colors[model],
            linewidth=0.5,
            edgecolor="black"# Add white edges to precision bars
        )

    # Plot precision bars (below x-axis)
    for i, model in enumerate(models):
        ax.bar(
            x + i * bar_width,
            metric2_data[model],
            width=bar_width,
            color=colors[model],
            linewidth=0.5,
            edgecolor="black" # Add white edges to precision bars
        )

    # Add a horizontal line at y=0
    ax.axhline(0, linewidth=0.8)

    # Customize y-axis ticks to show absolute values
    ax.set_ylim(-1, 1)
    y_ticks = np.linspace(-1, 1, 11)  # Generate ticks from -1 to 1
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{abs(tick):.1f}" for tick in y_ticks])  # Use absolute values for labels
    

    # Add labels for "Recall" and "Precision"
    ax.text(-2.4, 0.42, metric1_name, fontsize=14, ha='center', rotation=90)
    ax.text(-2.4, -0.63, metric2_name, fontsize=14, ha='center', rotation=90)

    # Customize the legend to show each model only once and make the title white
    custom_legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[model], label=model) for model in models
    ]
    legend = ax.legend(handles=custom_legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
    #legend.set_title("Model", prop={'size': 12, 'weight': 'bold', 'color': 'white'})  # Set title font size and color
    plt.setp(legend.get_texts())  # Set legend text color to white
    # Customize the plot
    ax.set_xticks(x + (len(models) - 1) * bar_width / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Adjust layout for better visibility
    plt.tight_layout()
    return fig, ax
