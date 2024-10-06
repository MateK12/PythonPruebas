import matplotlib.pyplot as plt
import pandas as pd
def plot_data(history):
    hist = pd.DataFrame(history.history)
    hist['epochs'] = history.epoch
    fig,err_plot =  plt.subplots()
    fig.suptitle('MAE error')
    err_plot.set_xlabel('Epoch')
    err_plot.set_ylabel('Error')
    err_plot.plot(hist['epochs'],hist['mae'], label='error en entrenamiento')
    err_plot.plot(hist['epochs'],hist['val_mae'], label='error en validacion')
    err_plot.legend()

    fig1,err_squered_plot = plt.subplots()
    err_squered_plot.set_xlabel('Epochs')
    err_squered_plot.set_xlabel('Error')
    fig1.suptitle('MSE error')
    err_squered_plot.plot(hist['epochs'],hist['mse'], label='error en entrenamiento')
    err_squered_plot.plot(hist['epochs'],hist['val_mse'], label='error en validacion')
    plt.show()

def plot_scatter(test_label, predictions):
    fig, ax = plt.subplots()
    fig.suptitle('Distribucion')
    ax.scatter(test_label, predictions)
    ax.set_xlabel('valores reales')
    ax.set_ylabel('valores predecidos')
    ax.axis('equal')
    ax.axis('square')
    plt.show()