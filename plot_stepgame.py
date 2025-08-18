import matplotlib.pyplot as plt
import numpy as np
import pickle

# Set font for publication quality
# Use sans-serif with fallback options for cross-platform compatibility
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
plt.rcParams['ps.fonttype'] = 42   # TrueType fonts for PostScript

# Iteration range
iterations = np.array([1, 2, 3])

# StepGame K=5 data
deepseek_k5_data = {
    'success': [91.7, 98.1, 100.0],
    'accuracy': [89.9, 94.5, 96.4]
}

gpt_k5_data = {
    'success': [93.1, 94.4, 95.8],
    'accuracy': [91.7, 91.7, 94.0]
}

llama_k5_data = {
    'success': [76.4, 87.5, 92.3],
    'accuracy': [75.0, 84.7, 87.5]
}

# StepGame K=10 data
deepseek_k10_data = {
    'success': [72.2, 91.7, 95.8],
    'accuracy': [59.7, 77.8, 82.9]
}

gpt_k10_data = {
    'success': [94.3, 98.1, 98.1],
    'accuracy': [83.0, 86.8, 86.8]
}

llama_k10_data = {
    'success': [64.2, 83.5, 88.7],
    'accuracy': [52.8, 67.9, 73.6]
}

# Organize all data
all_data = {
    'K5_success': {
        'deepseek': deepseek_k5_data['success'],
        'gpt': gpt_k5_data['success'],
        'llama': llama_k5_data['success']
    },
    'K5_accuracy': {
        'deepseek': deepseek_k5_data['accuracy'],
        'gpt': gpt_k5_data['accuracy'],
        'llama': llama_k5_data['accuracy']
    },
    'K10_success': {
        'deepseek': deepseek_k10_data['success'],
        'gpt': gpt_k10_data['success'],
        'llama': llama_k10_data['success']
    },
    'K10_accuracy': {
        'deepseek': deepseek_k10_data['accuracy'],
        'gpt': gpt_k10_data['accuracy'],
        'llama': llama_k10_data['accuracy']
    }
}

# Colors matching the SparQA style
colors = {
    'deepseek': '#00897B',   # Teal/green
    'gpt': '#FF6B35',        # Orange  
    'llama': '#7B68EE'       # Purple
}

# Markers for each model
markers = {
    'deepseek': 'o',  # Circle
    'gpt': 's',       # Square
    'llama': '^'      # Triangle
}

# Subplot configurations
subplot_configs = [
    ('K5_success', 'K=5 Success Rate', 'Program Success Rate (%)', [70, 105], range(70, 110, 10)),
    ('K5_accuracy', 'K=5 Accuracy Rate', 'Answer Accuracy Rate (%)', [70, 100], range(70, 105, 10)),
    ('K10_success', 'K=10 Success Rate', 'Program Success Rate (%)', [60, 105], range(60, 110, 10)),
    ('K10_accuracy', 'K=10 Accuracy Rate', 'Answer Accuracy Rate (%)', [50, 90], range(50, 95, 10))
]

# Create figure with single row of 4 plots
fig = plt.figure(figsize=(16, 6))

# Adjusted grid spec with better left margin for y-axis labels
gs = fig.add_gridspec(1, 4, hspace=0.0, wspace=0.20, 
                     left=0.06, right=0.98, top=0.90, bottom=0.18)

                   
fig.suptitle('Effects of Feedback Loop between LLM and ASP on StepGame Dataset',
             fontsize=13, fontweight='bold', y=0.96)

# Create subplots
axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

# Plot each subplot
for ax_idx, (ax, (data_key, title, y_label, y_lim, y_ticks)) in enumerate(zip(axes, subplot_configs)):
    
    # Get data for this subplot
    subplot_data = all_data[data_key]
    
    # Plot each model
    models_data = [
        ('deepseek', subplot_data['deepseek'], 'DeepSeek Chat'),
        ('gpt', subplot_data['gpt'], 'GPT-4o'),
        ('llama', subplot_data['llama'], 'Llama 3 70B Instruct')
    ]
    
    # Determine line style based on metric type
    if 'success' in data_key:
        line_style = '--'  # Dashed for success
    else:
        line_style = '-'   # Solid for accuracy
    
    # Store values for smart positioning
    all_values_iter1 = []
    all_values_iter3 = []
    
    for model_idx, (model_key, values, label) in enumerate(models_data):
        all_values_iter1.append(values[0])
        all_values_iter3.append(values[2])
        
        # Plot line
        line = ax.plot(iterations, values,
                      label=label,
                      color=colors[model_key],
                      marker=markers[model_key],
                      linestyle=line_style,
                      linewidth=2.0,
                      markersize=7,
                      markeredgewidth=1.5,
                      markeredgecolor='white',
                      zorder=2 + model_idx)
    
    # Add value labels for iterations 1 and 3 with smart positioning
    for model_idx, (model_key, values, label) in enumerate(models_data):
        
        # Iteration 1 labels
        x_pos = iterations[0]
        y_pos = values[0]
        
        # Calculate vertical offset based on value distribution
        sorted_idx_1 = sorted(range(len(all_values_iter1)), key=lambda k: all_values_iter1[k])
        pos_rank_1 = sorted_idx_1.index(model_idx)
        
        # Special handling for overlapping values
        value_diffs_1 = [abs(all_values_iter1[j] - y_pos) for j in range(len(all_values_iter1)) if j != model_idx]
        min_diff_1 = min(value_diffs_1) if value_diffs_1 else float('inf')
        
        if min_diff_1 < 2:  # Values too close
            y_offset_1 = [-4, 0, 4][pos_rank_1]
        else:
            y_offset_1 = [-2.5, 0, 2.5][pos_rank_1]
        
        ax.annotate(f'{y_pos:.1f}',
                   xy=(x_pos, y_pos),
                   xytext=(-10, y_offset_1),
                   textcoords='offset points',
                   fontsize=8,
                   ha='right',
                   va='center',
                   color=colors[model_key],
                   fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.2', 
                            facecolor='white', 
                            edgecolor='none',
                            alpha=0.7),
                   zorder=10)
        
        # Iteration 3 labels
        x_pos = iterations[2]
        y_pos = values[2]
        
        # Calculate vertical offset based on value distribution
        sorted_idx_3 = sorted(range(len(all_values_iter3)), key=lambda k: all_values_iter3[k])
        pos_rank_3 = sorted_idx_3.index(model_idx)
        
        # Special handling for overlapping values
        value_diffs_3 = [abs(all_values_iter3[j] - y_pos) for j in range(len(all_values_iter3)) if j != model_idx]
        min_diff_3 = min(value_diffs_3) if value_diffs_3 else float('inf')
        
        if min_diff_3 < 2:  # Values too close
            y_offset_3 = [-4, 0, 4][pos_rank_3]
        else:
            y_offset_3 = [-2.5, 0, 2.5][pos_rank_3]
        
        ax.annotate(f'{y_pos:.1f}',
                   xy=(x_pos, y_pos),
                   xytext=(10, y_offset_3),
                   textcoords='offset points',
                   fontsize=8,
                   ha='left',
                   va='center',
                   color=colors[model_key],
                   fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.2', 
                            facecolor='white', 
                            edgecolor='none',
                            alpha=0.7),
                   zorder=10)
    
    # Add K value as text annotation in the lower right corner of each plot
    k_value = 'K=5' if 'K5' in data_key else 'K=10'
    hops = '5 hops' if 'K5' in data_key else '10 hops'
    ax.text(0.98, 0.03, f'{k_value}\n({hops})', 
            transform=ax.transAxes,
            fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', 
                     edgecolor='gray',
                     alpha=0.8),
            fontweight='normal')
    
    # X-axis
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_xticks(iterations)
    ax.set_xlim(0.6, 3.4)
    
    # Y-axis label - simplified and closer to axis
    if ax_idx in [0, 2]:  # Success rate plots
        ax.set_ylabel('Program Success Rate (%)', fontsize=12, labelpad=2)
    else:  # Accuracy rate plots
        ax.set_ylabel('Answer Accuracy Rate (%)', fontsize=12, labelpad=2)
    
    # Set y-axis limits and ticks
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    
    # Grid styling - minimal and clean
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='gray')
    ax.set_facecolor('#FAFAFA')
    
    # Spines styling - clean academic look
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#333333')
    
    # Tick parameters
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(which='major', labelsize=9)
    ax.tick_params(axis='y', pad=1)  # Reduce padding between y-tick labels and axis

# Create a clean legend at the bottom
legend_elements = []
for model, label in zip(['deepseek', 'gpt', 'llama'],
                       ['DeepSeek Chat', 'GPT-4o', 'Llama 3 70B Instruct']):
    legend_elements.append(plt.Line2D([0], [0],
                                     color=colors[model],
                                     marker=markers[model],
                                     linestyle='-',
                                     linewidth=2,
                                     markersize=7,
                                     markeredgewidth=1.5,
                                     markeredgecolor='white',
                                     label=label))

# Add line style indicators
legend_elements.append(plt.Line2D([0], [0], color='dimgray', linestyle='--',
                                  linewidth=2, label='Logic Program Success Rate'))
legend_elements.append(plt.Line2D([0], [0], color='dimgray', linestyle='-',
                                  linewidth=2, label='Answer Accuracy Rate'))

# Place legend at the bottom center
fig.legend(handles=legend_elements,
          loc='lower center',
          ncol=5,
          fontsize=11,
          frameon=True,
          fancybox=False,
          shadow=False,
          edgecolor='#CCCCCC',
          facecolor='white',
          bbox_to_anchor=(0.5, 0.01),
          columnspacing=1.8,
          handlelength=2.2)

# Add a vertical separator line between K=5 and K=10 groups
fig.text(0.5, 0.45, '|', fontsize=80, color='#DDDDDD', ha='center', va='center', 
         transform=fig.transFigure, alpha=0.3)

# Fine-tune layout
plt.tight_layout(rect=[0.02, 0.12, 0.98, 0.92])

# Save in multiple formats for publication
print("Saving StepGame dataset figures (single row layout)...")

# Save as PDF (vector format, best for publications)
plt.savefig('stepgame_feedback_single_row_refined.pdf', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print("✓ PDF saved: stepgame_feedback_single_row_refined.pdf")

# Save as EPS (alternative vector format)
plt.savefig('stepgame_feedback_single_row_refined.eps', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("✓ EPS saved: stepgame_feedback_single_row_refined.eps")

# Save as high-resolution PNG
plt.savefig('stepgame_feedback_single_row_refined.png', dpi=600, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("✓ PNG saved: stepgame_feedback_single_row_refined.png (600 DPI)")

# Display the figure
plt.show()

print("\n" + "="*50)
print("StepGame dataset figure (refined) generated successfully!")
print("="*50)
print("\nKey improvements:")
print("- Fixed y-axis label positioning with proper labelpad")
print("- Reduced spacing between y-axis labels and ticks")
print("- Simplified y-axis label text")
print("- Better overall spacing and margins")
print("- Maintained publication-quality appearance")