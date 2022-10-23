import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np


def plotDistribution(data, test, exp, xlabel, res_dir=None):
    d = pd.DataFrame(columns=['Probabilities', 'Split'])
    d['Probabilities'] = np.concatenate((data, test), axis=0)[:,0]
    d['Split'] = np.concatenate((np.repeat('Training', data.shape[0]), np.repeat('Testing', test.shape[0])), axis=0)
    d['Probabilities'] = d['Probabilities']

    #plt.tight_layout()
    sns.displot(d, x="Probabilities", hue="Split", kind="kde", fill=True)
    #plt.hist(d["Probabilities"])
    # plt.xlim(0, 1)
    #plt.title(exp, fontsize=12, fontweight='bold', y=0.9)
    # axes[1,1].set_xlim(0, 1)
    plt.xlabel(xlabel, fontsize=10, fontweight='bold')
    plt.ylabel("Density", fontsize=10, fontweight='bold')
    if res_dir is not None:
        plt.savefig(res_dir + exp + ".png")
    else:
        plt.savefig(exp + ".png")
    plt.close()


def make_plot(values, samples, filename, n=64, top_bottom='top'):
    asort = np.argsort(values.reshape(values.shape[0]))
    if top_bottom == 'top':
        samples_to_plot = samples[asort[len(asort) - n:]][::-1]
    else:
        samples_to_plot = samples[asort[0:n]]

    row_col=int(np.sqrt(n))
    for i in range(0, n):
        plt.subplot(row_col, row_col, i + 1)
        plt.imshow(samples_to_plot[i], cmap=plt.get_cmap('gray'))
        plt.axis('off')
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename + "_" + top_bottom + ".png")
    plt.close()


def save_res(X, X_test, train_scores, test_scores, y_test, exp_name, file_out, num, out_dir, plot_top_bottom=True):
    auc_score = roc_auc_score(y_test, test_scores)
    print(auc_score)
    file_out.write(exp_name+": " + str(auc_score))

    plotDistribution(train_scores.reshape(train_scores.shape[0], 1), test_scores.reshape(test_scores.shape[0], 1),
                     exp_name + " " + str(num), "Probs", res_dir=out_dir)

    if plot_top_bottom:
        make_plot(train_scores, X, filename=out_dir+"Train_" + exp_name + "_" + str(num), top_bottom='top')
        make_plot(train_scores, X, filename=out_dir+"Train_" + exp_name + "_" + str(num), top_bottom='bottom')
        make_plot(test_scores, X_test, filename=out_dir+"Test_" + exp_name + "_" + str(num), top_bottom='top')
        make_plot(test_scores, X_test, filename=out_dir+"Test_" + exp_name + "_" + str(num), top_bottom='bottom')