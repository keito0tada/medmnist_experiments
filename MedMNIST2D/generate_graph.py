import os
import re
import pandas as pd

import matplotlib.pyplot as plt

num_points = 7
def tex_line_dfs(dfs, row, name):
    target = [0.0 for _ in range(num_points)]
    diff = [0.0 for _ in range(num_points)]
    for df in dfs:
        for i in range(num_points):
            target[i] += df.loc[row, i + 1]
            if i == 6:
                diff[i] += df.loc[row, i + 1] / df.loc['retain', i + 1] 
            else:
                diff[i] += df.loc[row, i + 1] - df.loc['retain', i + 1] 
    # return f'{name} & {target[0] / len(dfs):.2f} ({diff[0] / len(dfs):.2f})' \
    #     +  f'& {target[1] / len(dfs):.2f} ({diff[1] / len(dfs):.2f})' \
    #     +  f'& {target[2] / len(dfs):.2f} ({diff[2] / len(dfs):.2f})' \
    #     +  f'& {target[3] / len(dfs):.2f} ({diff[3] / len(dfs):.2f})' \
    #     +  f'& {target[4] / len(dfs):.0f} ({diff[4] / len(dfs):.0f})' \
    #     +  f'& {target[5] / len(dfs):.2f} ({diff[5] / len(dfs):.2f})' \
    #     +  f'& {target[6] / len(dfs):.0f} ({diff[6] / len(dfs):.0f})' \
    #     + r' \\' + '\n'
    return f'{name}' \
                + f'& {target[0] / len(dfs):.4f}' \
                + f'& {target[1] / len(dfs):.4f}' \
                + f'& {target[2] / len(dfs):.4f}' \
                + f'& {target[3] / len(dfs):.4f}' \
                + f'& {target[4] / len(dfs):.0f}' \
                + f'& {target[5] / len(dfs):.4f}' \
                + f'& {target[6] / len(dfs):.0f}' \
                + r' \\' + '\n' \
                + f'& ({diff[0] / len(dfs):.4f})' \
                + f'& ({diff[1] / len(dfs):.4f})' \
                + f'& ({diff[2] / len(dfs):.4f})' \
                + f'& ({diff[3] / len(dfs):.4f})' \
                + f'& ' \
                + f'& ' \
                + f'& ({diff[6] / len(dfs):.4f})' \
                + r' \\' + '\n'

# def tex_line(df, base, row, name):
#     return f'{name} & {df.loc[row][1]:.4f} & {df.loc[row][2]:.4f} & {df.loc[row][3]:.4f} & {df.loc[row][4]:.4f} & {df.loc[row][5]:.4f} & {df.loc[row][6]:.4f} & {df.loc[row][7]:.0f}' + r' \\' + '\n'

def get_avg_df(dfs):
    df = dfs[0]
    for index in df.index.values:
        for col in range(1, num_points + 1):
            for i in range(1, len(dfs)):
                df.loc[index, col] += dfs[i].loc[index, col]
            df.loc[index, col] = df.loc[index, col] / len(dfs)
    return df
        

def get_df(data_flag, rate, index, epochs):
    path_dataset = f'/data1/keito/bachelor/model/{data_flag}'
    path_model = os.path.join(path_dataset, f'retrain_{rate}_{index}')
    path_saliency = os.path.join(path_model, 'saliency')
    path_eval = os.path.join(path_model, 'eval.csv')
    path_target_time = os.path.join(path_dataset, f'target_{index}/{data_flag}_log.txt')
    path_retrain_time = os.path.join(path_model, f'{data_flag}_log.txt') 
    path_rl_time = os.path.join(path_model, 'RL_unlearn_time.txt')
    path_sal_time = os.path.join(path_saliency, f'RL_0.7_unlearn_time.txt')
    
    df = pd.read_csv(path_eval, index_col=0, header=None)

    df[7] = 0
    with open(path_target_time) as f:
        df.loc['target', 7] = float(f.readline()[6:])
    with open(path_retrain_time) as f:
        df.loc['retain', 7] = float(f.readline()[6:])
    with open(path_rl_time) as f:
        rl_times = {}
        times = f.readlines()
        for i in times:
            matcher = re.match(r'epoch: ([0-9]+), time: ([0-9.]+)', i).groups()
            rl_times[int(matcher[0])] = float(matcher[1])
        for epoch in epochs:
            df.loc[f'RL_{epoch}', 7] = rl_times[epoch]
    with open(path_sal_time) as f:
        rl_times = {}
        times = f.readlines()
        for i in times:
            matcher = re.match(r'epoch: ([0-9]+), time: ([0-9.]+)', i).groups()
            rl_times[int(matcher[0])] = float(matcher[1])
        for epoch in epochs:
            df.loc[f'RL_{epoch}_sal', 7] = rl_times[epoch] 
    return df

def main2(data_flag, rate, indexes, epochs):
    dfs = []
    for index in indexes:
        dfs.append(get_df(data_flag, rate, index, epochs))
    
    tex_table = r'モデル & TA & FA & RA & MIA & Wdist & Ddist & Time \\ \hline' + '\n' 
    tex_table += tex_line_dfs(dfs, 'target', '元のモデル')
    tex_table += tex_line_dfs(dfs, 'retain', '再訓練モデル')
    for epoch in epochs:
        tex_table += tex_line_dfs(dfs, f'RL_{epoch}', f'RL ({epoch} epochs)')
    for epoch in epochs:
        tex_table += tex_line_dfs(dfs, f'RL_{epoch}_sal', f'SalUn ({epoch} epochs)')
    print(tex_table)

# def main(data_flag, rate, index):
#     epochs = [5, 10]

#     path_dataset = f'/data1/keito/bachelor/model/{data_flag}'
#     path_model = os.path.join(path_dataset, f'retrain_{rate}_{index}')
#     path_saliency = os.path.join(path_model, 'saliency')
#     path_eval = os.path.join(path_model, 'eval.csv')
#     path_target_time = os.path.join(path_dataset, f'target_{index}/{data_flag}_log.txt')
#     path_retrain_time = os.path.join(path_model, f'{data_flag}_log.txt') 
#     path_rl_time = os.path.join(path_model, 'RL_unlearn_time.txt')
#     path_sal_time = os.path.join(path_saliency, f'RL_0.7_unlearn_time.txt')
    
#     df = pd.read_csv(path_eval, index_col=0, header=None)

#     df[7] = 0
#     with open(path_target_time) as f:
#         df.loc['target', 7] = float(f.readline()[6:])
#     with open(path_retrain_time) as f:
#         df.loc['retain', 7] = float(f.readline()[6:])
#     with open(path_rl_time) as f:
#         rl_times = {}
#         times = f.readlines()
#         for i in times:
#             matcher = re.match(r'epoch: ([0-9]+), time: ([0-9.]+)', i).groups()
#             rl_times[int(matcher[0])] = float(matcher[1])
#         for epoch in epochs:
#             df.loc[f'RL_{epoch}', 7] = rl_times[epoch]
#     with open(path_sal_time) as f:
#         rl_times = {}
#         times = f.readlines()
#         for i in times:
#             matcher = re.match(r'epoch: ([0-9]+), time: ([0-9.]+)', i).groups()
#             rl_times[int(matcher[0])] = float(matcher[1])
#         for epoch in epochs:
#             df.loc[f'RL_{epoch}_sal', 7] = rl_times[epoch]

#     tex_table = r'モデル & TA(%) & FA(%) & RA(%) & MIA(%) & Wdist & Odist & Time(s) \\ \hline' + '\n'
#     tex_table += tex_line(df, df.loc['retain'], 'target', '元のモデル')
#     tex_table += tex_line(df, df.loc['retain'], 'retain', '再訓練モデル')
#     for epoch in epochs:
#         tex_table += tex_line(df, df.loc['retain'], f'RL_{epoch}', f'RL ({epoch} epochs)')
#     for epoch in epochs:
#         tex_table += tex_line(df, df.loc['retain'], f'RL_{epoch}_sal', f'SalUn ({epoch} epochs)')
#     print(tex_table)

def plot_ax(ax, data_flag, rate, indexes, epochs, target='RL_{}_sal', x=7, y=1):
    dfs = []
    for index in indexes:
        dfs.append(get_df(data_flag, rate, index, epochs))
    df = get_avg_df(dfs)

    data = []
    for epoch in epochs:
        data.append((df.loc[target.format(epoch), x], df.loc[target.format(epoch), y]))
    data.sort()
    ax.plot([s for s, t in data], [t for s, t in data], label=f'{int(100 * rate)}%')


def gen_graph(data_flag, indexes):
    epochs = range(5, 51, 5)
    
    fig, axes = plt.subplots(1, 1)

    for i in [0.1, 0.3, 0.5]:
        plot_ax(axes, data_flag, i, indexes, epochs, x=7, y=1)

    axes.set_xlabel('Time')
    axes.set_ylabel('TA')
    axes.legend()

    plt.show()

gen_graph('pathmnist', [0])
gen_graph('octmnist', [0])
gen_graph('tissuemnist', [0])
# main2('pathmnist', 0.1, [0], [5, 10])
# main2('pathmnist', 0.3, [0], [5, 10])
# main2('pathmnist', 0.5, [0], [5, 10])
# main2('octmnist', 0.1, [0], [5, 10])
# main2('octmnist', 0.3, [0], [5, 10])
# main2('octmnist', 0.5, [0], [5, 10])
# main2('tissuemnist', 0.1, [0], [5, 10])
# main2('tissuemnist', 0.3, [0], [5, 10])
# main2('tissuemnist', 0.5, [0], [5, 10])
