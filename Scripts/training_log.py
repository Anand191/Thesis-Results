import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logfolder', help='Log data folder', default='../MicroTasks-Training')
parser.add_argument('--plotfolder', help='Plot data folder', default='../Plots')
opt = parser.parse_args()


fnames = ['gru_hard', 'lstm_hard'] #['gru_baseline', 'lstm_baseline'] 
#ftypes = ['np_longer', 'np_unseen', 'np_unseen_longer']
ftypes = ['longer', 'unseen', 'unseen_longer']
hyperparam_str = 'E128_H128'
final_fnames = []
for ftype in ftypes:
    for fname in fnames:
        final_fnames.append(fname+'_{}_{}_LOG'.format(ftype, hyperparam_str))

for fname in final_fnames:
    try:
        file_folder = fname.replace('_LOG','')
        df = pd.read_csv(os.path.join(opt.logfolder, fname), delim_whitespace=True)
        df = df.T
        df.fillna(0, inplace=True)
        holder_arr = np.zeros((df.shape[0]-1, df.shape[1]+1))
        holder_arr[:,0] = df.index[1:]
        holder_arr[:, 1:] = df.iloc[1:,:]
        col_names = ['steps']
        for n in df.iloc[0]:
            if ('/' in n):
                temp = n.split('/')[-1]
                temp2 = temp.split('.')[0]
                temp3 = temp2.split('_')[-1]
                col_names.append(temp3.lower())
            else:
                col_names.append(n.lower())
        new_df = pd.DataFrame(holder_arr, columns=col_names)
        dtypes = ['train', 'validation', 'longer', 'unseen']
        ndtypes = []
        indices = []
        for d in dtypes:
            if d in new_df.columns.values.tolist():
                ndtypes.append(d)
                indices.append(new_df.columns.values.tolist().index(d))
        dtypes = [x for _,x in sorted(zip(indices,ndtypes))]
        new_cols = ['steps']
        for d in dtypes:
            idx = new_df.columns.values.tolist().index(d)
            old_cols = new_df.columns.values.tolist()[idx+1:idx+5]
            temp_new_cols = [d+'_'+c for c in old_cols]
            new_cols.extend(temp_new_cols)
        new_df.drop(columns=dtypes, inplace=True)
        new_df.columns = new_cols
        losses = [l for l in new_df.columns.values.tolist() if 'loss' in l.split('_')]
        accuracies_all = [l for l in new_df.columns.values.tolist() if 'acc' in l.split('_') and 'erm' not in l.split('_')]
        accuracies = ['train_seq_acc', 'train_vp_tsk_acc', 'validation_seq_acc', 'validation_vp_tsk_acc', '_seq_acc', '_vp_tsk_acc']
        test_columns = list(set(accuracies_all) - set(accuracies))[0].split('_')[0]
        for i in range(-2,0):
            accuracies[i] = test_columns + accuracies[i]
        colours = ['r', 'b', 'g', 'm', 'k', 'y']
        folder_name = fname.split('_')[1]
        plt.figure()
        for i, acc in enumerate(accuracies):
            plt.plot(new_df['steps'], new_df[acc], color=colours[i], label=acc)
        plt.ylim(ymin=0, ymax=1.1)
        plt.yticks(np.arange(0, 1.1, 0.1))
        #new_df.plot(x='steps', y = accuracies)
        plt.legend(loc='best')
        plt.title('Accuracies_{}'.format(fname))
        plt.savefig(os.path.join(opt.plotfolder,fname.split('_')[1], fname+'_accuracies.png'))
        print('finished_plotting')

        # plt.figure()
        # new_df.plot(x='steps', y=losses)
        # plt.title('Loss_{}'.format(fname))
        # plt.savefig(os.path.join(opt.plotfolder, 'Hard', fname + '_loss.png'))
    except:
        import traceback
        traceback.print_exc()
