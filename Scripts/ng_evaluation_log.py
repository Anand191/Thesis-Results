import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logfolder', help='Log data folder', default='../MicroTasks-Evaluation')
parser.add_argument('--plotfolder', help='Plot data folder', default='../Plots')
opt = parser.parse_args()

def plot_bar(ops, base, hard, learn, name, title): #hard,
    a,b,c,d = title
    ind = np.arange(len(ops))
    width = 0.27
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind-width, base, width, color='r')
    rects2 = ax.bar(ind, learn, width, color='b')
    rects3 = ax.bar(ind + width, hard, width, color='g')


    ax.set_ylabel(name)
    ax.set_xticks(ind )#+ width/2)
    ax.set_xticklabels(ops)
    ax.legend((rects1[0], rects2[0], rects3[0] ), ('baseline','learned', 'hard')) #
    ax.set_title('{}_{}_{}_{}'.format(a,b,c,d))
    opt_path = os.path.join(opt.plotfolder,'New_AG','evaluate',title[1],title[2])
    if not os.path.exists(opt_path):
        os.makedirs(opt_path)
    plt.savefig(os.path.join(opt_path, '{}_{}.png'.format(a,name)))
    plt.close(fig)
    #plt.show()

cells = ['gru', 'lstm']
task_types = ['verify', 'produce']
ftypes = ['longer', 'unseen', 'unseen_longer']
acc_type = ['seq', 'vp' ]
for cell in cells:
    for task in task_types:
        if (task=='verify'):
            ops = ['and', 'or', 'not', 'copy']
        else:
            ops = ['and', 'or', 'not']
        for ftype in ftypes:
            df_base = pd.read_csv(os.path.join(opt.logfolder,'Ponderless', '{}_baseline_{}.tsv'.format(cell,task)), delimiter='\t', index_col=False)
            df_hard  = pd.read_csv(os.path.join(opt.logfolder, 'New_AG','{}_hard_{}.tsv'.format(cell, task)), delimiter='\t', index_col=False)
            df_learned = pd.read_csv(os.path.join(opt.logfolder, 'New_AG', '{}_learned_{}.tsv'.format(cell, task)), delimiter='\t', index_col=False)
            for acc in acc_type:
                base = []
                hard = []
                learned = []
                col_name = [c for c in df_base.columns.values.tolist() if acc in c.split('_')][0]
                for op in ops:
                    for i in range(df_base.shape[0]):
                        op_search = df_base['filename'].iloc[i].split('_')
                        f_search = op_search[1] if len(op_search)==3 else '_'.join(map(str,op_search[1:-1]))
                        if(op in op_search and ftype==f_search):
                            base.append(df_base[col_name].iloc[i])

                    for i in range(df_hard.shape[0]):
                        op_search = df_hard['filename'].iloc[i].split('_')
                        f_search = op_search[1] if len(op_search)==3 else '_'.join(map(str,op_search[1:-1]))
                        if(op in op_search and ftype==f_search):
                            hard.append(df_hard[col_name].iloc[i])

                    for i in range(df_learned.shape[0]):
                        op_search = df_learned['filename'].iloc[i].split('_')
                        f_search = op_search[1] if len(op_search)==3 else '_'.join(map(str,op_search[1:-1]))
                        if(op in op_search and ftype==f_search):
                            learned.append(df_learned[col_name].iloc[i])

                plot_bar(ops,base, hard, learned, col_name, (cell, task, ftype, acc))





