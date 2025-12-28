import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_schematic():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define boxes [x, y, width, height]
    # Current State
    ax.add_patch(patches.FancyBboxPatch((4, 5.0), 2, 0.8, boxstyle="round,pad=0.1", fc='lightblue', ec='black'))
    ax.text(5, 5.4, "Current State $x_t$", ha='center', va='center', fontsize=12)
    
    # Decision Node
    ax.add_patch(patches.Circle((5, 3.5), 0.4, fc='lightgray', ec='black'))
    ax.text(5, 3.5, "$u \\sim U$", ha='center', va='center', fontsize=10)
    
    # Arrow x -> Decision
    ax.annotate("", xy=(5, 3.9), xytext=(5, 5.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Global Branch (Left)
    ax.add_patch(patches.FancyBboxPatch((1, 1.5), 2.5, 1.0, boxstyle="round,pad=0.1", fc='#ffbfbf', ec='black'))
    ax.text(2.25, 2.0, "Flow Proposal\n$x' \\sim q_\\phi(x')$\n(ODE Integration)", ha='center', va='center', fontsize=10)
    
    # Arrow Decision -> Global
    ax.annotate("Global ($p_{glob}$)", xy=(2.25, 2.5), xytext=(4.6, 3.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=1.5))
    
    # Local Branch (Right)
    ax.add_patch(patches.FancyBboxPatch((6.5, 1.5), 2.5, 1.0, boxstyle="round,pad=0.1", fc='#bfffbf', ec='black'))
    ax.text(7.75, 2.0, "Local Proposal\n$x' \\sim K(x'|x_t)$", ha='center', va='center', fontsize=10)
    
    # Arrow Decision -> Local
    ax.annotate("Local ($1-p_{glob}$)", xy=(7.75, 2.5), xytext=(5.4, 3.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", lw=1.5))
    
    # MH Step
    ax.add_patch(patches.FancyBboxPatch((4, 0.2), 2, 0.8, boxstyle="round,pad=0.1", fc='wheat', ec='black'))
    ax.text(5, 0.6, "MH Acceptance\n$\\alpha(x_t, x')$", ha='center', va='center', fontsize=12)
    
    # Arrows to MH
    ax.annotate("", xy=(5, 1.0), xytext=(2.25, 1.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", lw=1.5))
    ax.annotate("", xy=(5, 1.0), xytext=(7.75, 1.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", lw=1.5))
    
    # Training Loop (Side box)
    ax.add_patch(patches.FancyBboxPatch((0.2, 3.5), 2.0, 1.5, boxstyle="square,pad=0.1", fc='#e0e0e0', ec='black', ls='--'))
    ax.text(1.2, 4.25, "Training Buffer\nSamples", ha='center', va='center', fontsize=10)
    ax.annotate("Train $\\phi$", xy=(1.5, 2.5), xytext=(1.2, 3.5), arrowprops=dict(arrowstyle="->", lw=1.5, ls='--'))

    plt.title("DiffMCMC Algorithm Schematic")
    plt.tight_layout()
    plt.savefig('paper/figures/schematic.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    draw_schematic()
