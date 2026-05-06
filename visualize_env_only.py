import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def generate_cliff_walking_map(filename='cliff_walking_map.png'):
    """Generate a clean visualization of the Cliff Walking environment."""
    # Environment Dimensions
    width = 12
    height = 4
    
    # Locations
    start_state = (3, 0)
    goal_state = (3, 11)
    cliff = [(3, i) for i in range(1, 11)]

    # Plot Setup
    fig, ax = plt.subplots(figsize=(14, 5))
    plt.rcParams.update({'font.size': 12})

    # Draw the grid
    for r in range(height):
        for c in range(width):
            state = (r, c)
            
            # Default style
            facecolor = '#E8F5E9'  # Light green for normal cells
            edgecolor = '#999'
            text = ""
            textcolor = '#333'
            fontweight = 'normal'

            if state in cliff:
                facecolor = '#D32F2F'  # Red for cliff
                text = "☠"
                textcolor = 'white'
            elif state == start_state:
                facecolor = '#4CAF50'  # Green for start
                text = "S"
                textcolor = 'white'
                fontweight = 'bold'
            elif state == goal_state:
                facecolor = '#FFD700'  # Gold for goal
                text = "G"
                textcolor = '#333'
                fontweight = 'bold'

            # Add cell
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5))
            
            # Add text label
            if text:
                ax.text(c, r, text, ha='center', va='center', fontsize=16,
                        color=textcolor, fontweight=fontweight)

    # Styling axes
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)  # Invert Y to match grid indexing
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_aspect('equal')
    ax.set_title('Cliff Walking Gridworld Environment', fontsize=18, fontweight='bold', pad=20)
    
    # Remove axis labels for a cleaner look
    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Rows', fontsize=12)

    # Add a legend
    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', label='Start (S)'),
        mpatches.Patch(facecolor='#FFD700', label='Goal (G)'),
        mpatches.Patch(facecolor='#D32F2F', label='Cliff (Death Zone)'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='#999', label='Normal Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.18, 1), 
              frameon=True, shadow=True, title="Legend")

    # Add grid text
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Successfully generated environment map: {filename}")

if __name__ == "__main__":
    generate_cliff_walking_map()
