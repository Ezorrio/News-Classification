import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, file_name):
        self.f = open("data/" + file_name, encoding="utf-8")
        self.reader = csv.reader(self.f)

    def getTrainTestData(self):
        self.f.seek(0)
        data = pd.read_csv(self.f)
        y = data.pop('category')
        x = data
        return train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)

    def getNewsCount(self):
        self.f.seek(0)
        return sum(1 for row in self.reader)

    def getCategories(self):
        categories = set()
        self.f.seek(0)
        for line in self.reader:
            current_category = line[1]
            if current_category == 'category':
                continue
            categories.add(current_category)
        return sorted(categories)

    def getCategoriesStats(self):
        stats = dict()
        self.f.seek(0)
        for line in self.reader:
            current_category = line[1]
            if current_category == 'category':
                continue
            if current_category in stats:
                stats[current_category] += 1
            else:
                stats[current_category] = 1
        return stats

    def showCategoriesStatsPlotMore(self, min_news_count):
        stats_all = self.getCategoriesStats()

        stats_slice = {k: stats_all[k] for k in stats_all.keys() if stats_all[k] > min_news_count}
        plt.bar(range(len(stats_slice)), list(stats_slice.values()), align='center')
        plt.xticks(range(len(stats_slice)), list(stats_slice.keys()))
        plt.xticks(fontsize=8, rotation=90)

        plt.show()

    def showCategoriesStatsPlotLess(self, max_news_count):
        stats_all = self.getCategoriesStats()

        stats_slice = {k: stats_all[k] for k in stats_all.keys() if stats_all[k] < max_news_count}
        plt.bar(range(len(stats_slice)), list(stats_slice.values()), align='center')
        plt.xticks(range(len(stats_slice)), list(stats_slice.keys()))
        plt.xticks(fontsize=8, rotation=90)

        plt.show()
