import os

models = ['simsiam']  # , 'simclr', 'barlowtwins')
datasets = ['Sentinel2GlobalLULC']
balanced_dataset = (False, True)
epochs = 25
batch_size = 64
ini_weights = 'random'
show_fig = False
cluster = True

# for s in range(0, times):
# print (f'Experiments: {times}')
for model in models:
    print(model)
    for dataset in datasets:
        print(dataset)
        for balanced in balanced_dataset:
            print(balanced)
            os.system(f'python3 03_1-PyTorch-Sentinel-2_SSL_SimSiam.py {model} '
                      f'--dataset {dataset} '
                      f'--balanced_dataset {balanced} '
                      f'--epochs {epochs} '
                      f'--batch_size {batch_size} '
                      f'--ini_weights {ini_weights} '
                      f'--show_fig {show_fig} '
                      f'--cluster {cluster}'
            )
