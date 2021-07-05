import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig

#%%
def plot_loss(l1, l2=None):

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(15,5))

    ax = plt.subplot(111)
    plt.plot(l1[0], c='blue', label=l1[1])
    if l2 is not None: plt.plot(l2[0], c='orange', label=l2[1])

    plt.ylabel('Loss')
    plt.ylim(0,1)

    plt.xlabel('Epoch')
    plt.xlim(0,max(len(l1[0]), len(l2[0])))

    plt.legend()

    sns.despine(offset=10, trim=True)

    plt.show()
    plt.close()


def plot_W(w):
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(15,5))

    ax = plt.subplot(121)
    mapp = plt.imshow(w,
                      # vmin=0, vmax=1,
                      cmap='coolwarm', #'viridis', #'Greys',
                      # interpolation='nearest'
                      )
    fig.colorbar(mapp, ax=ax)
    sns.despine(offset=10, trim=True)

    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    ax = plt.subplot(122)
    ew, _ = eig(w)

    # extract imaginary part
    x = [e.real for e in ew]

    # extract imaginary part
    y = [e.imag for e in ew]

    # mod_ew = [np.abs(e) for e in ew]

    plt.scatter(x,
                y,
                cmap='viridis'
                )
    plt.ylabel('Imaginary')
    plt.xlabel('Real')

    plt.xlim(-1,1)
    plt.ylim(-1,1)

    plt.subplots_adjust(wspace=0.4, hspace=None)

    sns.despine(offset=10, trim=True)

    plt.show()
    plt.close()


def plot_LE(les_stats, stat):

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(15,5))

    ax = plt.subplot(111)

    plt.plot(les_stats)
    plt.ylabel(f'{stat} LE')
    plt.xlabel('Epoch')

    max_epoch = len(les_stats)
    plt.xlim(0,max_epoch)
    plt.ylim(-6,6)

    sns.despine(offset=10, trim=True)

    plt.show()
    plt.close()


# def plot_property(property, title=None, xscaler=10, yscaler=10, scale_property=True, plot_grad=True, scale_grad=False):
#
#     if scale_property:
#         vmin = np.min(property)
#         vmax = np.max(property)
#         property = (property-vmin)/(vmax-vmin)
#
#     sns.set(style="ticks", font_scale=1.0)
#
#     height, width = property.shape
#     xscaler = int((1/3) * height) #20
#     yscaler = int((2/5) * width) #4
#
#     if plot_grad:
#         fig = plt.figure(figsize=(1.5*2.2*height/xscaler, 1.5*width/yscaler))
#         ax = plt.subplot(121)
#
#     else:
#         fig = plt.figure(figsize=(1.5*height/xscaler, 1.5*width/yscaler))
#         ax = plt.subplot(111)
#
#
#     # plot property evolution
#     mapp = plt.imshow(property.T,
#                       # extent=[None],
#                       aspect='auto',
#                       # vmin=0, vmax=1,
#                       cmap='viridis', #'Greys',
#                       # interpolation='nearest'
#                       )
#     fig.colorbar(mapp, ax=ax)
#     plt.ylabel('Neuron ID')
#     plt.xlabel('Epoch')
#
#     frame = plt.gca()
#     frame.axes.yaxis.set_ticklabels([])
#     frame.axes.yaxis.set_ticks([])
#
#     # plot gradient evolution
#     if plot_grad:
#         gradient = np.array([property[i]-property[i-1] for i in range(1,property.shape[0])])
#
#         if scale_grad:
#             vmin = np.min(gradient)
#             vmax = np.max(gradient)
#             gradient = (gradient-vmin)/(vmax-vmin)
#
#         ax = plt.subplot(122)
#
#         mapp = plt.imshow(gradient.T,
#                           # extent=[None],
#                           aspect='auto',
#                           # vmin=0, vmax=1,
#                           cmap='coolwarm', #'Greys',
#                           # interpolation='nearest'
#                           )
#
#         fig.colorbar(mapp, ax=ax)
#         plt.ylabel('Neuron ID')
#         plt.xlabel('Epoch')
#
#         frame = plt.gca()
#         frame.axes.yaxis.set_ticklabels([])
#         frame.axes.yaxis.set_ticks([])
#
#     if title is not None: plt.suptitle(title)
#
#     sns.despine(offset=10, trim=False, left=True)
#
#     frame1 = plt.gca()
#     frame1.axes.yaxis.set_ticklabels([])
#     frame1.axes.yaxis.set_ticks([])
#
#     plt.show()
#     plt.close()
#

def plot_property(property, plots, title=None, scale_property=True):

    if scale_property:
        vmin = np.min(property)
        vmax = np.max(property)
        property = (property-vmin)/(vmax-vmin)

    sns.set(style="ticks", font_scale=1.0)

    n_epochs, n_nodes = property.shape
    n_plots = len(plots)

    factor = 1.5
    xscaler = int((1/3) * n_epochs) #20
    yscaler = int((2/5) * n_nodes) #4

    fig = plt.figure(figsize=(factor*1.1*n_plots*n_epochs/xscaler, factor*n_nodes/yscaler))

    i = 1
    if 'value' in plots:
        ax = plt.subplot(1, n_plots, 1)

        # plot property evolution
        mapp = plt.imshow(property.T,
                          # extent=[None],
                          aspect='auto',
                          # vmin=0, vmax=1,
                          cmap='viridis', #'Greys',
                          # interpolation='nearest'
                          )
        fig.colorbar(mapp, ax=ax)
        plt.ylabel('Neuron ID')
        plt.xlabel('Epoch')

        frame = plt.gca()
        frame.axes.yaxis.set_ticklabels([])
        frame.axes.yaxis.set_ticks([])

        i+=1

    if 'grad' in plots:
        ax = plt.subplot(1, n_plots, i)

        gradient = np.array([property[i]-property[i-1] for i in range(1,property.shape[0])])

        mapp = plt.imshow(gradient.T,
                          # extent=[None],
                          aspect='auto',
                          # vmin=0, vmax=1,
                          cmap='coolwarm', #'Greys',
                          # interpolation='nearest'
                          )

        fig.colorbar(mapp, ax=ax)
        plt.ylabel('Neuron ID')
        plt.xlabel('Epoch')

        frame = plt.gca()
        frame.axes.yaxis.set_ticklabels([])
        frame.axes.yaxis.set_ticks([])

        i+=1

    if 'rank' in plots:
        ax = plt.subplot(1, n_plots, i)

        ranks = []
        for arr in property:
            temp = arr.argsort()
            rank = np.empty_like(temp)
            rank[temp] = np.arange(len(arr))
            ranks.append(rank)

        ranks = np.vstack(ranks)

        mapp = plt.imshow(ranks.T,
                          # extent=[None],
                          aspect='auto',
                          # vmin=0, vmax=1,
                          cmap='viridis', #'Greys',
                          # interpolation='nearest'
                          )

        fig.colorbar(mapp, ax=ax)
        plt.ylabel('Neuron ID')
        plt.xlabel('Epoch')

        frame = plt.gca()
        frame.axes.yaxis.set_ticklabels([])
        frame.axes.yaxis.set_ticks([])

    if title is not None: plt.suptitle(title)

    sns.despine(offset=10, trim=False, left=True)

    plt.show()
    plt.close()
