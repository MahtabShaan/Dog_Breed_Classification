import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import join
import os
import shutil

data_dir = 'C:/Users/Mahtab Noor Shaan/PycharmProjects/dog_breed_recognition'
labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))
print(len(listdir(join(data_dir, 'train'))), len(labels))
print(len(listdir(join(data_dir, 'test'))), len(sample_submission))

yy = pd.value_counts(labels['breed'])

fig, ax = plt.subplots()
fig.set_size_inches(15, 9)
sns.set_style("whitegrid")

ax = sns.barplot(x = yy.index, y = yy, data = labels)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 8)
ax.set(xlabel='Dog Breed', ylabel='Count')
ax.set_title('Distribution of Dog breeds')
plt.show()

train_path = 'C:/Users/Mahtab Noor Shaan/PycharmProjects/dog_breed_recognition/train/'
new_train_path = 'C:/Users/Mahtab Noor Shaan/PycharmProjects/dog_breed_recognition/new_train/'
#--- snippet to split train images into 120 folders ---

c = 0
for i in range(len(labels)):
    l = labels.id[i]
    for filename in os.listdir(train_path):
        f = filename[:-4]
        if (l == f):
            print(c)
            c+=1
            if not os.path.exists(new_train_path + labels.breed[i]):
                os.makedirs(new_train_path + labels.breed[i])
                shutil.copy2(train_path + filename, new_train_path + labels.breed[i])
            else:
                shutil.copy2(train_path + filename, new_train_path + labels.breed[i])