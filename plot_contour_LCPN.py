import pickle
import csv
from numpy.lib.arraypad import pad
import numpy as np
import matplotlib.pyplot as plt


# Classification boundary of three sources (engineered, natural, and incidental)
def plot_contour_1():
    with open(f'models/{hiclass}/MgP.Engineered.pickle', 'rb') as fmodel:
        classifier_Engineered = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Incidental.pickle', 'rb') as fmodel:
        classifier_Incidental = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Natural.pickle', 'rb') as fmodel:
        classifier_Natural = pickle.load(fmodel)

    X_Engineered = X[y[:, 1] == 'MgP.Engineered', :]
    X_Natural = X[y[:, 1] == 'MgP.Natural', :]
    X_Incidental = X[y[:, 1] == 'MgP.Incidental', :]

    X_test_Engineered = X_test[y_test[:, 1] == 'MgP.Engineered', :]
    X_test_Natural = X_test[y_test[:, 1] == 'MgP.Natural', :]
    X_test_Incidental = X_test[y_test[:, 1] == 'MgP.Incidental', :]

    f = plt.figure()
    ax = f.add_subplot(111)

    # Scatter plot
    plt.plot(X_Engineered[:, 0], X_Engineered[:, 1], 'o', color='#3351FF',
             markeredgecolor='k', label='EMNPs', alpha=0.7)
    plt.plot(X_Natural[:, 0], X_Natural[:, 1], 'o', color='#FFC033',
             markeredgecolor='k', label='NMNPs', alpha=0.7)
    plt.plot(X_Incidental[:, 0], X_Incidental[:, 1], 'o', color='#EA2F55',
             markeredgecolor='k', label='IMNPs', alpha=0.7)

    plt.plot(X_test_Engineered[:, 0], X_test_Engineered[:, 1], 's', color='#3351FF',
             markeredgecolor='k', label='EMNPs(ext)', alpha=0.7)
    plt.plot(X_test_Natural[:, 0], X_test_Natural[:, 1], 's', color='#FFC033',
             markeredgecolor='k', label='NMNPs(ext)', alpha=0.7)
    plt.plot(X_test_Incidental[:, 0], X_test_Incidental[:, 1], 's', color='#EA2F55',
             markeredgecolor='k', label='IMNPs(ext)', alpha=0.7)

    X_test_Natural = X_test_Troll[y_test_Troll[:, 1] == 'MgP.Natural', :]
    plt.plot(X_test_Natural[:, 0], X_test_Natural[:, 1], 's', color='#FFC033',
             markeredgecolor='k', alpha=0.7)

    plt.xlabel(u'$\delta^{18}$O$_{V-SMOW}$‰', fontsize=18)
    plt.ylabel(u'$\delta^{56}$Fe$_{IRMM014}$‰', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.set_ylim([0, 1.1])
    ax.set_xlim([-12, 25])
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z1_lp = classifier_Engineered.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z1_l = classifier_Engineered.predict(np.c_[xx.ravel(), yy.ravel()])
    Z2_lp = classifier_Incidental.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z2_l = classifier_Incidental.predict(np.c_[xx.ravel(), yy.ravel()])
    Z3_lp = classifier_Natural.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z3_l = classifier_Natural.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z1_l)

    # fill the color for each class region
    Z = np.zeros_like(Z1_l)
    Z[Z1_l == 1] = 0
    Z[Z2_l == 1] = 1
    Z[Z3_l == 1] = 2

    # Put the result into a color plot
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('mylist', ['#3351FF', '#EA2F55', '#FFC033'], N=3)

    # Plot Contour Line
    Z_Engineered = Z1_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_Engineered, [0.7], linewidths=2.5, alpha=0.8, colors='#3351FF')

    Z_Natural = Z3_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_Natural, [0.7], linewidths=2.5, alpha=0.8, colors='#FFC033')

    Z_Incidental = Z2_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_Incidental, [0.7], linewidths=2.5, alpha=0.8, colors='#EA2F55')

    legend = ax.legend(loc='upper right', frameon=False,
                       borderaxespad=0.1, handletextpad=0,
                       labelspacing=0.4, ncol=1, prop={'size': 14})
    legend.get_frame().set_facecolor('none')
    plt.tight_layout()
    plt.show()


# Classification boundary and predicted results of the chemical species of EMNPs (Fe3O4 and γ-Fe2O3) at the level 2
def plot_contour_2():
    with open(f'models/{hiclass}/MgP.Engineered.Mgh.pickle', 'rb') as fmodel:
        classifier_Mgh = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mag.pickle', 'rb') as fmodel:
        classifier_Mag = pickle.load(fmodel)

    X_mag = X[y[:, 2] == 'MgP.Engineered.Mag', :]
    X_mgh = X[y[:, 2] == 'MgP.Engineered.Mgh', :]
    print(X_mag)

    X_test_mag = X_test[y_test[:, 2] == 'MgP.Engineered.Mag', :]
    X_test_mgh = X_test[y_test[:, 2] == 'MgP.Engineered.Mgh', :]

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.plot(X_mag[:, 0], X_mag[:, 1], 'o', color='#3380FF',
             markeredgecolor='k', alpha=0.8, label=u'E-Fe$_3$O$_4$')
    plt.plot(X_mgh[:, 0], X_mgh[:, 1], 'o', color='#80BEE7',
             markeredgecolor='k', alpha=0.8, label=u'E-$\gamma$-Fe$_2$O$_3$')

    plt.plot(X_test_mag[:, 0], X_test_mag[:, 1], 's', color='#3380FF',
             markeredgecolor='k', alpha=0.8, label=u'E-Fe$_3$O$_4$(ext)')
    plt.plot(X_test_mgh[:, 0], X_test_mgh[:, 1], 's', color='#80BEE7',
             markeredgecolor='k', alpha=0.8, label=u'E-$\gamma$-Fe$_2$O$_3$(ext)')

    plt.xlabel(u'$\delta^{18}$O$_{V-SMOW}$‰', fontsize=18)
    plt.ylabel(u'$\delta^{56}$Fe$_{IRMM014}$‰', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    print(plt.xlim(), plt.ylim())
    ax.set_ylim([0, 0.8])
    ax.set_xlim([-12, 15])
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    Z1_lp = classifier_Mag.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z1_l = classifier_Mag.predict(np.c_[xx.ravel(), yy.ravel()])
    Z2_lp = classifier_Mgh.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z2_l = classifier_Mgh.predict(np.c_[xx.ravel(), yy.ravel()])

    # fill the color
    Z = np.zeros_like(Z1_l)
    Z[Z1_l == 1] = 0
    Z[Z2_l == 1] = 1

    # Plot Contour Line
    Z_mag = Z1_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_mag, [0.7], linewidths=2.5, alpha=0.8, colors='#3380FF')

    Z_mgh = Z2_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_mgh, [0.7], linewidths=2.5, alpha=0.8, colors='#80BEE7')

    legend = plt.legend(loc='upper right', frameon=False,
                        borderaxespad=0.1, handletextpad=0,
                        labelspacing=0.4, ncol=1, prop={'size': 14})
    legend.get_frame().set_facecolor('none')

    plt.tight_layout()
    plt.show()


# Classification boundary and predicted results of the synthetic method of E-γ-Fe2O3 (EP and ES) at the level 3
def plot_contour_3_mgh():
    with open(f'models/{hiclass}/MgP.Engineered.Mgh.EP.pickle', 'rb') as fmodel:
        classifier_MghEP = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mgh.ES.pickle', 'rb') as fmodel:
        classifier_MghES = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mag.EP.pickle', 'rb') as fmodel:
        classifier_MagEP = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mag.ES.pickle', 'rb') as fmodel:
        classifier_MagES = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mag.EA.pickle', 'rb') as fmodel:
        classifier_MagEA = pickle.load(fmodel)

    # L4
    X_MghEP = X[(y[:, 3] == 'MgP.Engineered.Mgh.EP'), :]
    X_MghES = X[(y[:, 3] == 'MgP.Engineered.Mgh.ES'), :]

    X_test_MghEP = X_test[(y_test[:, 3] == 'MgP.Engineered.Mgh.EP'), :]
    X_test_MghES = X_test[(y_test[:, 3] == 'MgP.Engineered.Mgh.ES'), :]

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.plot(X_MghEP[:, 0], X_MghEP[:, 1], 'o', color='#0099FF',
             markeredgecolor='k', alpha=0.8, label=u'E-$\gamma$-Fe$_2$O$_3$-EP')
    plt.plot(X_MghES[:, 0], X_MghES[:, 1], 'o', color='#00CC66',
             markeredgecolor='k', alpha=0.8, label=u'E-$\gamma$-Fe$_2$O$_3$-ES')

    plt.plot(X_test_MghEP[:, 0], X_test_MghEP[:, 1], 's', color='#0099FF',
             markeredgecolor='k', alpha=0.8, label=u'E-$\gamma$-Fe$_2$O$_3$-EP(ext)')
    plt.plot(X_test_MghES[:, 0], X_test_MghES[:, 1], 's', color='#00CC66',
             markeredgecolor='k', alpha=0.8, label=u'E-$\gamma$-Fe$_2$O$_3$-ES(ext)')

    plt.xlabel(u'$\delta^{18}$O$_{V-SMOW}$‰', fontsize=18)
    plt.ylabel(u'$\delta^{56}$Fe$_{IRMM014}$‰', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    print(plt.xlim(), plt.ylim())
    ax.set_ylim([0, 0.8])
    ax.set_xlim([-12, 15])
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    plt.savefig('111.png')

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z1_lp = classifier_MghEP.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z1_l = classifier_MghEP.predict(np.c_[xx.ravel(), yy.ravel()])
    Z2_lp = classifier_MghES.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z2_l = classifier_MghES.predict(np.c_[xx.ravel(), yy.ravel()])

    # fill the color
    Z = np.zeros_like(Z1_l)
    Z[Z1_l == 1] = 0
    Z[Z2_l == 1] = 1


    # Plot Contour Line
    Z_MghEP = Z1_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_MghEP, [0.7], linewidths=2.5, alpha=1, colors='#0099FF')

    Z_MghES = Z2_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_MghES, [0.7], linewidths=2.5, alpha=0.8, colors='#00CC66')

    legend = plt.legend(loc='upper right', frameon=False,
                        borderaxespad=0.1, handletextpad=0,
                        labelspacing=0.4, ncol=1, prop={'size': 14})
    legend.get_frame().set_facecolor('none')
    plt.tight_layout()
    plt.show()


# Classification boundary and predicted results of the synthetic method of engineered (E)-Fe3O4 (EP, ES, and EA) at the level 3
def plot_contour_3_mag():
    with open(f'models/{hiclass}/MgP.Engineered.Mgh.EP.pickle', 'rb') as fmodel:
        classifier_MghEP = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mgh.ES.pickle', 'rb') as fmodel:
        classifier_MghES = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mag.EP.pickle', 'rb') as fmodel:
        classifier_MagEP = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mag.ES.pickle', 'rb') as fmodel:
        classifier_MagES = pickle.load(fmodel)
    with open(f'models/{hiclass}/MgP.Engineered.Mag.EA.pickle', 'rb') as fmodel:
        classifier_MagEA = pickle.load(fmodel)

    # L4
    X_MghEP = X[(y[:, 3] == 'MgP.Engineered.Mgh.EP'), :]
    X_MagEP = X[(y[:, 3] == 'MgP.Engineered.Mag.EP'), :]
    X_MagES = X[(y[:, 3] == 'MgP.Engineered.Mag.ES'), :]
    X_MagEA = X[(y[:, 3] == 'MgP.Engineered.Mag.EA'), :]
    print(X_MghEP)

    X_test_MagEP = X_test[(y_test[:, 3] == 'MgP.Engineered.Mag.EP'), :]
    X_test_MagES = X_test[(y_test[:, 3] == 'MgP.Engineered.Mag.ES'), :]
    X_test_MagEA = X_test[(y_test[:, 3] == 'MgP.Engineered.Mag.EA'), :]

    f = plt.figure()
    ax = f.add_subplot(111)

    plt.plot(X_MagEP[:, 0], X_MagEP[:, 1], 'o', color='#0099FF',
             markeredgecolor='k', alpha=0.8, label=u'E-Fe$_3$O$_4$-EP')
    plt.plot(X_MagES[:, 0], X_MagES[:, 1], 'o', color='#00CC66',
             markeredgecolor='k', alpha=0.8, label=u'E-Fe$_3$O$_4$-ES')
    plt.plot(X_MagEA[:, 0], X_MagEA[:, 1], 'o', color='#A266FF',
             markeredgecolor='k', alpha=0.8, label=u'E-Fe$_3$O$_4$-EA')

    plt.plot(X_test_MagEP[:, 0], X_test_MagEP[:, 1], 's', color='#0099FF',
             markeredgecolor='k', alpha=0.8, label=u'E-Fe$_3$O$_4$-EP(ext)')
    plt.plot(X_test_MagES[:, 0], X_test_MagES[:, 1], 's', color='#00CC66',
             markeredgecolor='k', alpha=0.8, label=u'E-Fe$_3$O$_4$-ES(ext)')
    plt.plot(X_test_MagEA[:, 0], X_test_MagEA[:, 1], 's', color='#A266FF',
             markeredgecolor='k', alpha=0.8, label=u'E-Fe$_3$O$_4$-EA(ext)')

    plt.xlabel(u'$\delta^{18}$O$_{V-SMOW}$‰', fontsize=18)
    plt.ylabel(u'$\delta^{56}$Fe$_{IRMM014}$‰', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    print(plt.xlim(), plt.ylim())
    ax.set_ylim([0, 0.8])
    ax.set_xlim([-12, 15])
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    plt.savefig('111.png')

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    Z3_lp = classifier_MagEP.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z3_l = classifier_MagEP.predict(np.c_[xx.ravel(), yy.ravel()])
    Z4_lp = classifier_MagES.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z4_l = classifier_MagES.predict(np.c_[xx.ravel(), yy.ravel()])
    Z5_lp = classifier_MagEA.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z5_l = classifier_MagEA.predict(np.c_[xx.ravel(), yy.ravel()])

    # fill the color
    Z = np.zeros_like(Z3_l)
    Z[Z3_l == 1] = 2
    Z[Z4_l == 1] = 3
    Z[Z5_l == 1] = 4


    # Plot Contour Line####
    Z_MagEP = Z3_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_MagEP, [0.7], linewidths=2.5, alpha=0.7, colors='#0099FF')

    Z_MagES = Z4_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_MagES, [0.7], linewidths=2.5, alpha=0.7, colors='#00CC66')

    Z_MagEA = Z5_lp[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z_MagEA, [0.7], linewidths=2.5, alpha=0.7, colors='#A266FF')

    legend = plt.legend(loc='upper right', frameon=False,
                        borderaxespad=0.1, handletextpad=0,
                        labelspacing=0.4, ncol=1, prop={'size': 14})
    legend.get_frame().set_facecolor('none')
    plt.tight_layout()
    plt.show()


def list_to_array(lists, pad_value=0, max_length=None):
    if max_length is None:
        max_length = max(map(len, lists))
    padded_lists = [pad(l, (0, max_length - len(l)), mode='constant', constant_values=pad_value) for l in lists]
    return np.vstack(padded_lists)


hiclass = 'LCPN'

# load traning set X,y
X = np.loadtxt('data/X_train.txt', delimiter='\t')
y = []
with open('data/y_train_hiclass.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        y.append(row)
y = np.array(y)
y_flat = np.loadtxt('data/y_train_flat.txt', dtype='str')

# load test set X,y
X_test = np.loadtxt('data/X_test.txt', delimiter='\t')
y_test = []
with open('data/y_test_hiclass.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        y_test.append(row)
y_test = list_to_array(y_test)

# load test set X,y
X_test_Troll = np.loadtxt('data/X_test_Troll.txt', delimiter='\t')
y_test_Troll = []
with open('data/y_test_Troll_hiclass.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        y_test_Troll.append(row)
y_test_Troll = list_to_array(y_test_Troll)

plot_contour_1()
plot_contour_2()
plot_contour_3_mgh()
plot_contour_3_mag()
