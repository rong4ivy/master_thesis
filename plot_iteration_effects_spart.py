import matplotlib.pyplot as plt
import numpy as np
import pickle

# Set font for publication quality
# Use sans-serif with fallback options for cross-platform compatibility
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.9
plt.rcParams['xtick.major.width'] = 0.9
plt.rcParams['ytick.major.width'] = 0.9
# plt.rcParams['pdf.fonttype'] = 45  # TrueType fonts for PDF
# plt.rcParams['ps.fonttype'] = 45  # TrueType fonts for PostScript

# Trimmed iteration range (starts from 1 to emphasize feedback impact)
iterations = np.array([1, 2, 3])

# DeepSeek data (trimmed)
deepseek_data = {
    'FR': {'success': [76.3, 96.9, 100.0], 'accuracy': [60.0, 70.3, 77.8]},
    'FB': {'success': [66.2, 83.5, 90.9], 'accuracy': [62.5, 75.4, 82.6]},
    'YN': {'success': [94.2, 100.0, 100.0], 'accuracy': [76.5, 82.3, 82.3]},
    'CO': {'success': [50.9, 72.5, 87.0], 'accuracy': [45.3, 56.8, 75.4]}
}

# GPT-4o data (corrected name)
gpt_data = {
    'FR': {'success': [64.7, 77.5, 85.3], 'accuracy': [46.5, 60.8, 65.4]},
    'FB': {'success': [70.2, 83.6, 85.0], 'accuracy': [61.2, 71.6, 75.5]},
    'YN': {'success': [95.3, 100.0, 100.0], 'accuracy': [68.3, 77.9, 77.9]},
    'CO': {'success': [74.3, 80.2, 87.5], 'accuracy': [58.2, 62.5, 74.8]}
}

# LLaMA3 70B Instruct data
llama_data = {
    'FR': {'success': [74.8, 80.6, 86.9], 'accuracy': [48.7, 54.5, 59.7]},
    'FB': {'success': [58.2, 78.5, 83.4], 'accuracy': [56.8, 73.5, 80.4]},
    'YN': {'success': [81.3, 92.9, 96.5], 'accuracy': [66.3, 74.2, 76.4]},
    'CO': {'success': [66.4, 78.2, 85.3], 'accuracy': [52.5, 62.5, 68.5]}
}

# Colors matching your original style - exact colors from your image
colors = {
    'deepseek': '#00897B',   # Teal/green matching your original
    'gpt': '#FF6B35',        # Orange matching your original  
    'llama': '#7B68EE'       # Purple matching your original
}

# Markers for each model
markers = {
    'deepseek': 'o',  # Circle
    'gpt': 's',       # Square
    'llama': '^'      # Triangle
}

# Question types
question_types = ['FR', 'FB', 'YN', 'CO']
qt_full_names = {
    'FR': 'Find Relation',
    'FB': 'Find Block',
    'YN': 'Yes/No',
    'CO': 'Choose Object'
}

# Create figure with two rows for Success and Accuracy (keeping original layout)
# Increased height from 8 to 10, reduced wspace from 0.25 to 0.12
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 4, hspace=0.15, wspace=0.12, 
                      left=0.06, right=0.98, top=0.90, bottom=0.11)
# Add main title with more space above subplots
fig.suptitle('Effects of Feedback Loop between LLM and ASP on SparQA Dataset',
             fontsize=13, fontweight='bold', y=0.99)

# Create subplots
axes_success = [fig.add_subplot(gs[0, i]) for i in range(4)]
axes_accuracy = [fig.add_subplot(gs[1, i]) for i in range(4)]

# Plotting function with refined value labels
def plot_metric(axes_list, metric_type, y_label):
    for i, qt in enumerate(question_types):
        ax = axes_list[i]
        
        # Plot each model
        models_data = [
            ('deepseek', deepseek_data, 'DeepSeek Chat'),
            ('gpt', gpt_data, 'GPT-4o'),
            ('llama', llama_data, 'Llama 3 70B Instruct')
        ]
        
        # Store values for smart positioning
        all_values_iter1 = []
        all_values_iter3 = []
        
        for model_idx, (model_key, data, label) in enumerate(models_data):
            if metric_type == 'success':
                values = data[qt]['success']
                line_style = '--'  # Dashed for success (matching your original)
            else:  # accuracy
                values = data[qt]['accuracy']
                line_style = '-'   # Solid for accuracy (but could be dashed too based on legend)
            
            all_values_iter1.append(values[0])
            all_values_iter3.append(values[2])
            
            # Plot line with exact style from original
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
        for model_idx, (model_key, data, label) in enumerate(models_data):
            values = data[qt][metric_type] if metric_type == 'success' else data[qt]['accuracy']
            
            # Iteration 1 labels
            x_pos = iterations[0]
            y_pos = values[0]
            
            # Calculate vertical offset based on value distribution
            sorted_idx_1 = sorted(range(len(all_values_iter1)), key=lambda k: all_values_iter1[k])
            pos_rank_1 = sorted_idx_1.index(model_idx)
            
            if len(set(all_values_iter1)) == 1:  # All same value
                y_offset_1 = [-3, 0, 3][model_idx]
            else:
                # Spread labels based on ranking
                y_offset_1 = [-3, 0, 3][pos_rank_1]
            
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
            
            if len(set(all_values_iter3)) == 1:  # All same value
                y_offset_3 = [-3, 0, 3][model_idx]
            else:
                # Spread labels based on ranking
                y_offset_3 = [-3, 0, 3][pos_rank_3]
            
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
        
        # Subplot title - only for success rate plots (top row)
        if metric_type == 'success':
            ax.set_title(f"{qt_full_names[qt]}\n({qt})", 
                        fontsize=10, fontweight='bold', pad=8)
        # No title for accuracy rate plots (bottom row)
        
        # X-axis
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_xticks(iterations)
        ax.set_xlim(0.6, 3.4)
        
        # Y-axis label only for leftmost plots
        if i == 0:
            ax.set_ylabel(y_label, fontsize=12)
        
        # Set y-axis limits based on metric type
        if metric_type == 'success':
            ax.set_ylim(45, 105)
            ax.set_yticks(range(50, 105, 10))
        else:
            ax.set_ylim(40, 85)
            ax.set_yticks(range(40, 90, 10))
        
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

# Plot success rates (top row)
plot_metric(axes_success, 'success', 'Program Success Rate (%)')

# Plot accuracy rates (bottom row)
plot_metric(axes_accuracy, 'accuracy', 'Accuracy Rate (%)')


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
          fontsize=12,
          frameon=True,
          fancybox=False,
          shadow=False,
          edgecolor='#CCCCCC',
          facecolor='white',
          bbox_to_anchor=(0.5, 0.01),
          columnspacing=2.0,
          handlelength=2.5)

# Fine-tune layout
plt.tight_layout(rect=[0.03, 0.07, 0.99, 0.95])

# Save in multiple formats for publication
print("Saving publication-ready figures...")

# Save as PDF (vector format, best for publications)
plt.savefig('sparqa_feedback_enhanced.pdf', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print("✓ PDF saved: sparqa_feedback_enhanced.pdf")

# # Save as EPS (alternative vector format)
# plt.savefig('sparqa_feedback_enhanced.eps', dpi=300, bbox_inches='tight',
#            facecolor='white', edgecolor='none')
# print("✓ EPS saved: sparqa_feedback_enhanced.eps")

# Save as high-resolution PNG
plt.savefig('sparqa_feedback_enhanced.png', dpi=600, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("✓ PNG saved: sparqa_feedback_enhanced.png (600 DPI)")

# # Save as SVG (vector format that can be edited)
# plt.savefig('sparqa_feedback_enhanced.svg', bbox_inches='tight',
#            facecolor='white', edgecolor='none')
# print("✓ SVG saved: sparqa_feedback_enhanced.svg")

# # Save as pickle file (.fig equivalent for Python/matplotlib)
# with open('sparqa_feedback_enhanced.fig', 'wb') as f:
#     pickle.dump(fig, f)
# # print("✓ FIG saved: sparqa_feedback_enhanced.fig (Python pickle format)")

# # Create MATLAB-compatible data structure
# matlab_data = {
#     'iterations': iterations.tolist(),
#     'deepseek_data': deepseek_data,
#     'gpt_data': gpt_data,
#     'llama_data': llama_data,
#     'colors': colors,
#     'question_types': question_types,
#     'qt_full_names': qt_full_names
# }

# # Save data for MATLAB import
# import json
# with open('sparqa_data_for_matlab.json', 'w') as f:
#     json.dump(matlab_data, f, indent=2)
# print("✓ MATLAB data saved: sparqa_data_for_matlab.json")

# # Display the figure
# plt.show()

print("\n" + "="*50)
print("Publication-ready figure generated successfully!")
print("="*50)
print("\nKey improvements:")
print("- Clean value labels for iterations 1 and 3 only")
print("- Smart positioning to avoid overlap")
print("- Two-row layout maintained (SUCCESS RATE and ACCURACY RATE)")
print("- Professional academic styling")
print("- Multiple export formats for journal submission")
print("\nFor journal submission:")
print("- Use PDF or EPS for vector graphics")
print("- PNG at 600 DPI for raster requirements")
print("- All files maintain exact style from original code")