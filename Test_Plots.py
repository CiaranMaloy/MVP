# Plot python

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle as pkl
import Test_Functions as test

file_loc = 'Files/'
filenames = test.get_files(file_loc)


save_loc = 'Saved_Figures'
test.make_folder(save_loc)

# clean filenames: (only .pkl files)
filenames_clean = []
for file in filenames:
    if file.endswith('.pkl'):
        filenames_clean.append(file)
print(filenames_clean)

# read pkl data 
# -- maybe you can have a list of objects, idk 
data_list = []
for file in filenames_clean:
    with open(file, 'rb') as f:
        data = pkl.load(f)
    data_list.append(data)

for data in data_list:

    # real data
    ones_real = []
    for point in range(len(data['True Cut Points'])):
        ones_real.extend([1])

    ones_pred = []
    for point in range(len(data['Predicted Cut Points'])):
        ones_pred.extend([1])

    name = data['Album Name']
    name = name.split('/')
    name = name[-2]

    plt.figure(figsize=(20, 4))
    plt.title(name)
    plt.xlabel('Split Location (minutes)')
    plt.ylabel('Indicator (0/1)')
    plt.stem(np.array(data['Predicted Cut Points'])/60, ones_pred)
    markerline, stemlines, baseline = plt.stem(
        np.array(data['True Cut Points'])/60, ones_real, linefmt='r', markerfmt='Dr', bottom=0, use_line_collection=True)
    markerline.set_markerfacecolor('none')
    plt.legend(['Predictions', 'Real'])
    plt.savefig(save_loc + '/' + name + '.png')
    plt.show()