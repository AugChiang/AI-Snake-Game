from IPython import display
import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf()) # get curretn figure
    plt.clf() # clear current figure
    plt.title("Training Process")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='mean score')
    plt.legend()
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)