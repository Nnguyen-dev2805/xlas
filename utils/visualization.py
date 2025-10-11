"""
Visualization Tools
==================

Các công cụ visualization cho báo cáo đồ án
- Plot comparison grids
- Histogram visualization
- Kernel visualization
- Interactive plots

Author: Image Processing Team
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_image_grid(images, titles, rows=None, cols=None, figsize=(15, 10), 
                   cmap='gray', save_path=None):
    """
    Hiển thị grid các ảnh
    
    Args:
        images: List các ảnh
        titles: List các title
        rows, cols: Số hàng và cột
        figsize: Kích thước figure
        cmap: Colormap
        save_path: Đường dẫn lưu file
        
    Returns:
        fig: Figure object
    """
    n_images = len(images)
    
    if rows is None and cols is None:
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
    elif rows is None:
        rows = (n_images + cols - 1) // cols
    elif cols is None:
        cols = (n_images + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_images):
        if len(images[i].shape) == 3:
            axes[i].imshow(images[i])
        else:
            axes[i].imshow(images[i], cmap=cmap)
        
        axes[i].set_title(titles[i], fontsize=10)
        axes[i].axis('off')
    
    # Ẩn axes thừa
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_histogram_comparison(histograms, labels, colors=None, figsize=(12, 6),
                            save_path=None):
    """
    So sánh nhiều histogram
    
    Args:
        histograms: List các histogram
        labels: List các label
        colors: List các màu
        figsize: Kích thước figure
        save_path: Đường dẫn lưu file
        
    Returns:
        fig: Figure object
    """
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(histograms)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(256)
    
    for i, (hist, label, color) in enumerate(zip(histograms, labels, colors)):
        ax.plot(x, hist, label=label, color=color, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Intensity Level')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_kernel_visualization(kernels, kernel_names, figsize=(15, 10), 
                            save_path=None):
    """
    Visualization các kernel
    
    Args:
        kernels: List các kernel
        kernel_names: List tên kernel
        figsize: Kích thước figure
        save_path: Đường dẫn lưu file
        
    Returns:
        fig: Figure object
    """
    n_kernels = len(kernels)
    cols = min(4, n_kernels)
    rows = (n_kernels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        im = axes[i].imshow(kernel, cmap='RdBu', interpolation='nearest')
        axes[i].set_title(f'{name}\nSum: {np.sum(kernel):.3f}')
        
        # Thêm text annotations
        if kernel.shape[0] <= 7:  # Chỉ hiển thị text cho kernel nhỏ
            for ki in range(kernel.shape[0]):
                for kj in range(kernel.shape[1]):
                    text_color = 'white' if abs(kernel[ki, kj]) > 0.5 * np.max(np.abs(kernel)) else 'black'
                    axes[i].text(kj, ki, f'{kernel[ki, kj]:.2f}', 
                               ha='center', va='center', color=text_color, fontsize=8)
        
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Ẩn axes thừa
    for i in range(n_kernels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_processing_pipeline(original, intermediate_results, final_result,
                           step_names, figsize=(20, 12), save_path=None):
    """
    Visualization pipeline xử lý ảnh
    
    Args:
        original: Ảnh gốc
        intermediate_results: List các kết quả trung gian
        final_result: Kết quả cuối cùng
        step_names: Tên các bước
        figsize: Kích thước figure
        save_path: Đường dẫn lưu file
        
    Returns:
        fig: Figure object
    """
    n_steps = len(intermediate_results) + 2  # +2 for original and final
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, n_steps, height_ratios=[3, 1])
    
    # Ảnh gốc
    ax0 = fig.add_subplot(gs[0, 0])
    if len(original.shape) == 3:
        ax0.imshow(original)
    else:
        ax0.imshow(original, cmap='gray')
    ax0.set_title('Original', fontsize=12, fontweight='bold')
    ax0.axis('off')
    
    # Các bước trung gian
    for i, (result, name) in enumerate(zip(intermediate_results, step_names)):
        ax = fig.add_subplot(gs[0, i+1])
        if len(result.shape) == 3:
            ax.imshow(result)
        else:
            ax.imshow(result, cmap='gray')
        ax.set_title(name, fontsize=12)
        ax.axis('off')
        
        # Thêm arrow
        if i < len(intermediate_results) - 1:
            ax.annotate('', xy=(1.1, 0.5), xytext=(0.9, 0.5),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Kết quả cuối cùng
    ax_final = fig.add_subplot(gs[0, -1])
    if len(final_result.shape) == 3:
        ax_final.imshow(final_result)
    else:
        ax_final.imshow(final_result, cmap='gray')
    ax_final.set_title('Final Result', fontsize=12, fontweight='bold', color='red')
    ax_final.axis('off')
    
    # Thêm thống kê ở hàng dưới
    for i in range(n_steps):
        ax_stats = fig.add_subplot(gs[1, i])
        
        if i == 0:
            img = original
            title = 'Original Stats'
        elif i == n_steps - 1:
            img = final_result
            title = 'Final Stats'
        else:
            img = intermediate_results[i-1]
            title = f'Step {i} Stats'
        
        # Tính thống kê
        stats_text = f"""
        Shape: {img.shape}
        Min: {np.min(img)}
        Max: {np.max(img)}
        Mean: {np.mean(img):.1f}
        Std: {np.std(img):.1f}
        """
        
        ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='center')
        ax_stats.set_title(title, fontsize=10)
        ax_stats.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_metrics(metrics_data, method_names, figsize=(12, 8),
                          save_path=None):
    """
    Visualization metrics so sánh
    
    Args:
        metrics_data: Dictionary chứa metrics
        method_names: Tên các phương pháp
        figsize: Kích thước figure
        save_path: Đường dẫn lưu file
        
    Returns:
        fig: Figure object
    """
    n_metrics = len(metrics_data)
    n_methods = len(method_names)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[i]
        
        # Bar plot
        bars = ax.bar(method_names, values, alpha=0.7)
        
        # Thêm value labels trên bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(metric_name)
        ax.set_ylabel('Value')
        
        # Rotate x labels if needed
        if max(len(name) for name in method_names) > 8:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_comparison(images, titles, figsize=(15, 10)):
    """
    Tạo comparison plot tương tác
    
    Args:
        images: List các ảnh
        titles: List các title
        figsize: Kích thước figure
        
    Returns:
        fig: Figure object
    """
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_images):
        if len(images[i].shape) == 3:
            im = axes[i].imshow(images[i])
        else:
            im = axes[i].imshow(images[i], cmap='gray')
        
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        
        # Thêm colorbar cho grayscale images
        if len(images[i].shape) == 2:
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Ẩn axes thừa
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    return fig


def save_all_visualizations(results_dict, output_dir, prefix=""):
    """
    Lưu tất cả visualization
    
    Args:
        results_dict: Dictionary chứa kết quả
        output_dir: Thư mục output
        prefix: Prefix cho tên file
        
    Returns:
        saved_files: List các file đã lưu
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_files = []
    
    # Lưu image grid
    if 'images' in results_dict and 'titles' in results_dict:
        fig = plot_image_grid(results_dict['images'], results_dict['titles'])
        filename = os.path.join(output_dir, f"{prefix}image_grid.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
    
    # Lưu histogram comparison
    if 'histograms' in results_dict and 'hist_labels' in results_dict:
        fig = plot_histogram_comparison(results_dict['histograms'], 
                                      results_dict['hist_labels'])
        filename = os.path.join(output_dir, f"{prefix}histogram_comparison.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
    
    # Lưu kernel visualization
    if 'kernels' in results_dict and 'kernel_names' in results_dict:
        fig = plot_kernel_visualization(results_dict['kernels'], 
                                      results_dict['kernel_names'])
        filename = os.path.join(output_dir, f"{prefix}kernels.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
    
    return saved_files
