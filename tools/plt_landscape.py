# --*-- conding:utf-8 --*--
# @time:9/18/25 02:12
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_landscape.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch

# Grid
xs = np.linspace(-3.0, 3.0, 240)
ys = np.linspace(-3.0, 3.0, 240)
X, Y = np.meshgrid(xs, ys)

def quantum_landscape_multi(X, Y):
    Z = 0.08*(X**2 + 0.6*Y**2) + 0.2*np.sin(0.6*np.sqrt(X**2 + Y**2))
    centers = [(-1.8, -1.2), (1.6, -1.0), (0.2, 1.8), (-0.2, 0.4)]
    depths = [2.4, 2.0, 1.8, 1.2]
    widths = [0.55, 0.70, 0.60, 0.45]
    for (cx, cy), d, w in zip(centers, depths, widths):
        Z -= d*np.exp(-(((X-cx)**2 + (Y-cy)**2)/(2*w**2)))
    return Z, centers

def dl_landscape_patches(X, Y, centers):
    Z = np.zeros_like(X)
    P = np.zeros_like(X)
    for (cx, cy) in centers:
        r2 = (X-cx)**2 + (Y-cy)**2
        patch = np.exp(-r2/(2*0.55**2))
        P = np.maximum(P, patch)
        hf = 0.45*np.sin(4.5*(X-cx)) * np.cos(4.5*(Y-cy)) + 0.22*np.sin(7*(X+0.3-cx)+2.5*(Y-0.2-cy))
        motif = -0.9*np.exp(-r2/0.11) - 0.55*np.exp(-((X-(cx+0.5))**2 + (Y-(cy-0.4))**2)/0.16)
        Z += patch*(hf + motif)
    return Z, P

def clean_axes(ax):
    # keep ticks but hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # hide axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    # keep grid and panes visible
    ax.grid(True)
    ax.xaxis.pane.set_visible(True)
    ax.yaxis.pane.set_visible(True)
    ax.zaxis.pane.set_visible(True)


if __name__ == '__main__':


    # Landscapes
    Zq, centers = quantum_landscape_multi(X, Y)
    Zdl, P = dl_landscape_patches(X, Y, centers)

    # Figure 1: Quantum landscape (multiple basins)
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Zq, cmap="Blues", linewidth=0, antialiased=True)
    ax1.set_title("Quantum Energy Landscape")
    clean_axes(ax1)
    ax1.legend(handles=[Patch(facecolor=plt.get_cmap("Blues")(0.7), edgecolor='none', label="Quantum landscape")],
               loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig("./mnt/energy_landscape_quantum_multi.pdf", bbox_inches="tight")
    plt.savefig("./mnt/energy_landscape_quantum_multi.svg", bbox_inches="tight")
    plt.show()

    # Figure 2: DL landscape (local patches near quantum basins)
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, Zdl, cmap="Reds", linewidth=0, antialiased=True)
    ax2.set_title("Deep-Learning Statistical Potentials")
    clean_axes(ax2)
    ax2.legend(handles=[Patch(facecolor=plt.get_cmap("Reds")(0.7), edgecolor='none', label="DL statistical potentials (local)")],
               loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig("./mnt/energy_landscape_dl_patches.pdf", bbox_inches="tight")
    plt.savefig("./mnt/energy_landscape_dl_patches.svg", bbox_inches="tight")
    plt.show()

    # Figure 3: Fused landscape (DL-detailed valleys; Quantum elsewhere)
    alpha, beta = 1.0, 0.95
    Zf_valley = alpha*Zq + beta*Zdl
    threshold_patch = 0.28
    threshold_energy = np.quantile(Zf_valley, 0.50)
    valley_mask = (P > threshold_patch) & (Zf_valley <= threshold_energy)
    Zmix = np.where(valley_mask, Zf_valley, Zq)

    cmp_q = plt.get_cmap("Blues")
    cmp_d = plt.get_cmap("Reds")
    colors = np.zeros(Zmix.shape + (4,), dtype=float)
    colors[valley_mask] = cmp_d(0.70)
    colors[~valley_mask] = cmp_q(0.70)

    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(X, Y, Zmix, facecolors=colors, linewidth=0, antialiased=True)
    ax3.set_title("Fused Energy Landscape")
    clean_axes(ax3)
    ax3.legend(handles=[
        Patch(facecolor=cmp_d(0.70), edgecolor='none', label="DL-detailed valleys"),
        Patch(facecolor=cmp_q(0.70), edgecolor='none', label="Quantum elsewhere"),
    ], loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig("./mnt/energy_landscape_fused_valleyMasked.pdf", bbox_inches="tight")
    plt.savefig("./mnt/energy_landscape_fused_valleyMasked.svg", bbox_inches="tight")
    plt.show()
