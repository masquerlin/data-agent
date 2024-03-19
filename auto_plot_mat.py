import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# df = pd.read_csv("100条虚拟数据20240223.csv")
# df = df[['业务单元','单价/费率暂估总价(万元)','首选中标金额(万元)','投标人数量', '备选中标金额(万元)']]
# df = df[['业务单元','单价/费率暂估总价(万元)']]
# df = df[['资审开标时间','资审结束时间','公告时间','标段审核通过时间','首选中标金额(万元)']]
class plot_data:
    def __init__(self) -> None:
        pass
    def plot_to_numpy(self, figure):
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        return img_array
    def delete_ax(self, fig, axes):
        
        print(axes)
        if not isinstance(axes, plt.Axes):
            if len(axes.shape) == 1:
                for i in range(axes.size):
                    ax = axes[i]
                    if not any([len(ax.lines), len(ax.collections), len(ax.patches)]):
                        print('shanchu')
                        fig.delaxes(ax)
            else:
                for i in range(axes.shape[0]):
                    for j in range(axes.shape[1]):
                        ax = axes[i,j]
                        if not any([len(ax.lines), len(ax.collections), len(ax.patches)]):
                            print('shanchu')
                            fig.delaxes(ax)


    def auto_plot(self, df):
        time_cols = [col for col in df.columns if ('时间' in col or 'time' in col) and df[col].dtype != 'datetime64[ns]']
        print(f"time_cols*****************{time_cols}")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果没有SimHei，也可以替换为其他支持中文的字体名
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方框的问题
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        category_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        
        for col in time_cols:
            print(df[col].dtype)
            if col in numeric_cols and col :
                try:
                    # 尝试转换列数据为datetime类型
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = df[col].dt.strftime('%Y-%m-%d')
                    df[col] = pd.to_datetime(df[col],  format='%Y-%m-%d')
                except Exception as e:
                    print(f"无法将列 '{col}' 转换为 datetime 类型：{str(e)}")
        time_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
        time_index = pd.api.types.is_datetime64_any_dtype(df.index)
        numeric_cols = [x for x in numeric_cols if x not in time_cols]
        # 初始化numpy数组图像列表
        image_arrays = []

        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            df[numeric_cols].hist(ax=ax)
            ax.set_title('数值列分布')
            # 将直方图转换为numpy数组并添加至列表
            image_arrays.append(self.plot_to_numpy(fig))

        if len(numeric_cols) > 1:
            print(numeric_cols)
            # 确定子图网格大小，假设是1行3列（根据实际情况调整）
            ncols = min(len(numeric_cols), 3)  # 如果你想一行最多显示3个图表
            nrows = (len(numeric_cols) - 1) // ncols + 1  # 计算需要几行才能放下所有图表
            print(nrows)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))  # 创建网格布局的figure和axes集合
            print(axes)
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
                        ax.set_title(f'{numeric_cols[i]} vs {numeric_cols[j]} 散点图')
                        ax.set_xlabel(numeric_cols[i])
                        ax.set_ylabel(numeric_cols[j])
                        if col < ncols - 1:
                            col = col+1
                        else:
                            col = 0
                            if row < nrows -1 :
                                row = row+1
                    except Exception as e:
                        print(e)
                        print(f"无法绘制{numeric_cols[i]}与{numeric_cols[j]}之间的箱线图：{str(e)}")
                    # 将散点图转换为numpy数组并添加至列表
            self.delete_ax(fig=fig, axes=axes)
            image_arrays.append(self.plot_to_numpy(fig))

        if len(category_cols) > 0 and len(numeric_cols) > 0:
            print(len(category_cols), len(numeric_cols))
            number = len(category_cols) * len(numeric_cols)
            ncols = min(number, 3)  # 如果你想一行最多显示3个图表
            nrows = (number - 1) // ncols + 1  # 计算需要几行才能放下所有图表
            row, col = 0, 0
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
            print(axes)
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
                        ax.set_title(f'按{cat_col}分类的{num_col}分布')
                        ax.set_xlabel(cat_col)
                        ax.set_ylabel(num_col)
                        if col < ncols - 1:
                            col = col+1
                        else:
                            col = 0
                            if row < nrows -1 :
                                row = row+1
                        # 将箱线图转换为numpy数组并添加至列表
                    except Exception as e:
                        print(e)
                        print(f"无法绘制{cat_col}与{num_col}之间的箱线图：{str(e)}")
            
            self.delete_ax(fig=fig, axes=axes)
            image_arrays.append(self.plot_to_numpy(fig))
                
        if (time_index or len(time_cols) > 0 ) and len(numeric_cols) > 0:
            if time_index:
                df = df.reset_index()
                df.rename(columns={'index': 'Timestamp'}, inplace=True)
                time_cols.append('Timestamp')
            number = len(time_cols) * len(numeric_cols)
            ncols = min(number, 3)  # 如果你想一行最多显示3个图表
            nrows = (number - 1) // ncols + 1  # 计算需要几行才能放下所有图表
            
            print(f"nrows*********{nrows}")
            print(f"nrows*********{ncols}")
            row, col = 0, 0
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
            # print(f"axes.size************{axes.size}")
            # print(f"axes.shape************{axes.shape}")
            print(axes)
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
                    print(f"time_col*****************{time_col}")
                    print(f"time_cols*****************{time_cols}")
                    print(f"numeric_cols*****************{numeric_cols}")
                    df.plot(kind='line', x=time_col, y=numeric_cols, ax=ax)
                    
                    ax.set_title(f'按{time_col}时间的{numeric_cols}的变化值')
                    ax.set_xlabel('日期')
                    ax.set_ylabel('数值')
                    if col < ncols - 1:
                        col = col+1
                    else:
                        col = 0
                        if row < nrows -1 :
                            row = row+1
                    # 将箱线图转换为numpy数组并添加至列表
                except Exception as e:
                    print(e)
                    print(f"无法按{time_col}时间的{numeric_cols}的变化值进行绘图，错误如下：{str(e)}")
            self.delete_ax(fig=fig, axes=axes)
            image_arrays.append(self.plot_to_numpy(fig))
        # 返回所有图片的numpy数组列表
        return image_arrays