from src.config import Configurations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

class PlotsGeneration():
    def __init__(self):
        self.configs = Configurations()
    
    def change_configs(self, new_configs: dict):
        self.configs.change_configs(new_configs)

    def plot_scores(self, f1_scores: pd.DataFrame):
        plt.figure()
        plt.plot(f1_scores['month'], f1_scores['f1_score'], '*-')
        plt.xlabel("year-month")
        plt.ylabel("f1 - score")

    def join_month_data(self):
        self.numerical_feats = self.configs.get_dict()["NUMERIC_FEAT"]
        eval_months = self.configs.get_dict()['EVAL_MONTHS']
        eval_months += [self.configs.get_dict()['TRAINING_MONTH']]
        self.eval_months = set(eval_months)
        global_data = pd.DataFrame()
        for month in self.eval_months:
            eval_data = pd.read_parquet(
                f"{self.configs.get_dict()['PROC_DATA_DIR']}/processed_data_{month}.parquet")
            eval_data["eval_month"] = month
            global_data = pd.concat((eval_data, global_data))
        self.global_data = global_data
            
    def generate_histograms(self):
        self.join_month_data()

        months = sorted(self.global_data['eval_month'].unique())

        n_feats = len(self.numerical_feats)
        n_months = len(months)

        fig = plt.figure(figsize=(5 * n_months, 3.5 * n_feats))
        outer = matplotlib.gridspec.GridSpec(n_feats * 2, n_months, height_ratios=[0.3, 3] * n_feats)

        for i, feature in enumerate(self.numerical_feats):
            title_ax = plt.Subplot(fig, outer[i * 2, :])
            title_ax.axis('off')
            title_ax.set_title(feature, fontsize=14, weight='bold')
            fig.add_subplot(title_ax)

            for j, month in enumerate(months):
                ax = plt.Subplot(fig, outer[i * 2 + 1, j])
                subset = self.global_data[self.global_data['eval_month'] == month]
                data = subset[feature].dropna()

                if data.nunique() <= 5:
                    sns.histplot(data, ax=ax, stat='count', discrete=True, color='skyblue')
                else:
                    q1 = data.quantile(0.25)
                    q3 = data.quantile(0.75)
                    iqr = q3 - q1
                    upper_limit = q3 + 1.5 * iqr
                    under_limit = q3 - 1.5 * iqr
                    filtered = data[(data <= upper_limit) & (data >= under_limit)]

                    if filtered.empty:
                        ax.set_visible(False)
                    else:
                        min_val = filtered.min()
                        max_val = filtered.max()
                        bins = min(30, filtered.nunique())
                        sns.histplot(filtered, ax=ax, stat='count', bins=bins, binrange=(min_val, max_val), color='skyblue')

                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title(month, fontsize=10)
                fig.add_subplot(ax)

        plt.tight_layout()
        plt.show()
    
    def generate_boxplots(self):
        self.join_month_data()

        months = sorted(self.global_data['eval_month'].unique())

        n_feats = len(self.numerical_feats)
        n_months = len(months)

        fig = plt.figure(figsize=(5 * n_months, 3.5 * n_feats))
        outer = matplotlib.gridspec.GridSpec(n_feats * 2, n_months, height_ratios=[0.3, 3] * n_feats)

        for i, feature in enumerate(self.numerical_feats):
            data_all = self.global_data[feature]
            if data_all.nunique() <= 10:
                continue

            title_ax = plt.Subplot(fig, outer[i * 2, :])
            title_ax.axis('off')
            title_ax.set_title(feature, fontsize=14, weight='bold')
            fig.add_subplot(title_ax)

            for j, month in enumerate(months):
                ax = plt.Subplot(fig, outer[i * 2 + 1, j])
                subset = self.global_data[self.global_data['eval_month'] == month]
                data = subset[feature].dropna()

               
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_limit = q1 - 5 * iqr # visualize some outliers
                upper_limit = q3 + 5 * iqr

                sns.boxplot(x=data, ax=ax, color='skyblue', width=0.5, fliersize=2)
                ax.set_xlim(lower_limit, upper_limit)

                ax.set_title(month, fontsize=10)
                fig.add_subplot(ax)

        plt.tight_layout()
        plt.show()
