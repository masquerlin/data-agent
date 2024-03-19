import io, random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class plot_data:
    def __init__(self) -> None:
        pass
    def plot_to_numpy(self, figure):
        """
        Convert a Matplotlib figure to a numpy array representing an image.

        Args:
            figure (matplotlib.figure.Figure): Matplotlib figure to convert.

        Returns:
            numpy.array: Numpy array representing the image.
        """
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        return img_array
    def delete_ax(self, fig, axes):
        """
        Delete axes from a figure based on whether they contain any plots.

        Args:
            fig (matplotlib.figure.Figure): Matplotlib figure.
            axes (matplotlib.axes.Axes): Axes to check and potentially delete.
        """
        if not isinstance(axes, plt.Axes):
            if len(axes.shape) == 1:
                for i in range(axes.size):
                    ax = axes[i]
                    if not any([len(ax.lines), len(ax.collections), len(ax.patches)]):
                        fig.delaxes(ax)
            else:
                for i in range(axes.shape[0]):
                    for j in range(axes.shape[1]):
                        ax = axes[i,j]
                        if not any([len(ax.lines), len(ax.collections), len(ax.patches)]):
                            fig.delaxes(ax)

    def generate_random_colors(self, num_colors):
        colors = []
        for _ in range(num_colors):
            r = random.random()
            g = random.random()
            b = random.random()
            colors.append((r, g, b))
        return colors
    def auto_plot(self, df):
        """
        Automatically generate plots for numeric and categorical columns in a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing data to plot.

        Returns:
            list: List of numpy arrays representing generated plots.
        """
        time_cols = [col for col in df.columns if ('时间' in col or 'time' in col) and df[col].dtype != 'datetime64[ns]']
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        category_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        
        for col in time_cols:
            print(df[col].dtype)
            if col in numeric_cols and col :
                try:
                    
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = df[col].dt.strftime('%Y-%m-%d')
                    df[col] = pd.to_datetime(df[col],  format='%Y-%m-%d')
                except Exception as e:
                    print(f"can't transfer '{col}' into datetime type, error:{str(e)}")
        time_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
        time_index = pd.api.types.is_datetime64_any_dtype(df.index)
        numeric_cols = [x for x in numeric_cols if x not in time_cols]
        image_arrays = []

        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            df[numeric_cols].hist(ax=ax)
            ax.set_title('Distribution of Numeric Columns')
            image_arrays.append(self.plot_to_numpy(fig))

        if len(numeric_cols) > 1 :
            ncols = min(len(numeric_cols), 3)  
            nrows = (len(numeric_cols) - 1) // ncols + 1  
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8)) 
            row, col = 0, 0
            for i in range(len(numeric_cols)-1):
                for j in range(i+1, len(numeric_cols)):
                    try:
                        if nrows == 1:
                            if ncols == 1:
                                ax = axes
                            else:
                                ax = axes[col]
                        else:
                            ax = axes[row, col]
                        df.plot(kind = 'scatter', x = numeric_cols[i], y = numeric_cols[j], ax=ax)
                        for k, row in df.iterrows():
                            ax.annotate(f'({row[df.columns[i]]}, {row[df.columns[j]]})', (row[df.columns[i]], row[df.columns[j]]))
                        df.plot(kind = 'line', x = numeric_cols[i], y = numeric_cols[j], ax=ax)
                        ax.set_title(f'{numeric_cols[i]} vs {numeric_cols[j]} Scatter Plot')
                        ax.set_xlabel(numeric_cols[i])
                        ax.set_ylabel(numeric_cols[j])
                        if col < ncols - 1:
                            col = col+1
                        else:
                            col = 0
                            if row < nrows -1 :
                                row = row+1
                    except Exception as e:
                        print(f"Unable to plot scatter plot for {numeric_cols[i]} vs {numeric_cols[j]}: {str(e)}")
                    
            self.delete_ax(fig=fig, axes=axes)
            image_arrays.append(self.plot_to_numpy(fig))

        if len(category_cols) > 0 and len(numeric_cols) > 0:
            number = len(category_cols) * len(numeric_cols)
            ncols = min(number, 3)  
            nrows = (number - 1) // ncols + 1 
            row, col = 0, 0
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
            for cat_col in category_cols:
                for num_col in numeric_cols:
                    try:
                        if nrows == 1:
                            if ncols == 1:
                                ax = axes
                            else:
                                ax = axes[col]
                        else:
                            ax = axes[row, col]
                        df.plot(kind= 'bar', x=cat_col, y=num_col, ax=ax)
                        df.plot(kind= 'line', x=cat_col, y=num_col, ax=ax)
                        for p in ax.patches:
                            ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                        ax.set_title(f'{num_col} Distribution by {cat_col}')
                        ax.set_xlabel(cat_col)
                        ax.set_ylabel(num_col)
                        if col < ncols - 1:
                            col = col+1
                        else:
                            col = 0
                            if row < nrows -1 :
                                row = row+1
                    except Exception as e:
                        print(f"Unable to plot bar chart for {cat_col} and {num_col}: {str(e)}")
            
            self.delete_ax(fig=fig, axes=axes)
            image_arrays.append(self.plot_to_numpy(fig))
                
        if (time_index or len(time_cols) > 0 ) and len(numeric_cols) > 0:
            if time_index:
                df = df.reset_index()
                df.rename(columns={'index': 'Timestamp'}, inplace=True)
                time_cols.append('Timestamp')
            number = len(time_cols) * len(numeric_cols)
            ncols = min(number, 3)  
            nrows = (number - 1) // ncols + 1  
            row, col = 0, 0
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
            for time_col in time_cols:
                try:
                    print(row, col)
                    ax = axes
                    if nrows == 1:
                        if ncols == 1:
                            ax = axes
                        else:
                            ax = axes[col]
                    else:
                        ax = axes[row, col]
                    df.plot(kind='line', x=time_col, y=numeric_cols, ax=ax)
                    for column in numeric_cols:
                        for i, value in enumerate(df[column]):
                            ax.text(i+1, value, str(value), ha='center', va='bottom')
                    ax.set_title(f'{numeric_cols} Trends Over {time_col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Numeric Values')
                    if col < ncols - 1:
                        col = col+1
                    else:
                        col = 0
                        if row < nrows -1 :
                            row = row+1
                except Exception as e:
                    print(f"Unable to plot trends for {numeric_cols} over {time_col}: {str(e)}")
            self.delete_ax(fig=fig, axes=axes)
            image_arrays.append(self.plot_to_numpy(fig))
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            row, col = 0, 0
            sum_value = []
            try:
                for col_name in numeric_cols:
                    # add the column sum
                    sum_value.append(df[col_name].sum())
                    # create pie
                print(sum_value)
                print(numeric_cols)
                num_colors = len(numeric_cols)
                colors = self.generate_random_colors(num_colors)
                wedges, texts, autotexts = ax.pie(sum_value, labels=[None]*len(numeric_cols), autopct='%1.1f%%', startangle=90, colors=colors)
                ax.set_title(f'Sum of {col_name}')

                # add point line
                bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="None", lw=0.72)
                kw = dict(arrowprops=dict(arrowstyle="-"),
                        bbox=bbox_props, zorder=0, va="center")

                for i, p in enumerate(wedges):
                    ang = (p.theta2 - p.theta1)/2. + p.theta1
                    y = np.sin(np.deg2rad(ang))
                    x = np.cos(np.deg2rad(ang))
                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                    connectionstyle = f"angle,angleA=0,angleB={ang}"
                    kw["arrowprops"].update({"connectionstyle": connectionstyle})
                    ax.annotate(numeric_cols[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                                horizontalalignment=horizontalalignment, **kw)
                ax.legend(wedges, numeric_cols, title="Columns", loc="lower right", bbox_to_anchor=(1, 0, 0.5, 1))    
            except Exception as e:
                print(f"Unable to plot pie chart for {col_name}: {str(e)}")
            image_arrays.append(self.plot_to_numpy(fig))
        return image_arrays