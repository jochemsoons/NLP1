import numpy as np
import matplotlib.pyplot as plt
import os

from utils import pload


if __name__ == '__main__': 
    fig, ax = plt.subplots()
    labels = ['0-10', '10-20', '20-30', '30+']
    x = np.arange(len(labels))  # the label locations
    width = 0.15
    models = ['BOW', 'CBOW', 'DeepCBOW', 'pt_DeepCBOW', 'LSTM', 'TreeLSTM']
    for i, model in enumerate(models):
        scores = np.asarray(pload('scores_{}_test_only'.format(model)))
        mean_scores = np.mean(scores, axis=0)
        std_devs = np.std(scores, axis=0)
        if i == 0:
            rects = ax.bar(x - width/2 - 2*width, mean_scores, width, label=model)
        elif i == 1:
            rects = ax.bar(x - width/2 - 1*width, mean_scores, width, label=model)
        elif i == 2:
            rects = ax.bar(x - width/2, mean_scores, width, label=model)
        elif i == 3:
            rects = ax.bar(x + width/2, mean_scores, width, label=model)
        elif i == 4:
            rects = ax.bar(x + width/2 + 1*width, mean_scores, width, label=model)
        elif i == 5:
            rects = ax.bar(x + width/2 + 2*width, mean_scores, width, label=model)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Sentence length')
    ax.set_title('Performance comparison for different sentence lengths')
    ax.set_xticks(x, labels)
    fig.tight_layout()
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower left', ncol=2)
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig('./plots/sentence_length')

