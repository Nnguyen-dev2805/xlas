"""
Comparison Tools
===============

Các công cụ so sánh kết quả
- Manual vs Library comparison
- Performance benchmarking
- Quality metrics
- Statistical analysis

Author: Image Processing Team
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any


def compare_implementations(image, func_manual, func_library, *args, **kwargs):
    """
    So sánh implementation manual vs library
    
    Args:
        image: Input image
        func_manual: Hàm manual implementation
        func_library: Hàm library implementation
        *args, **kwargs: Arguments cho các hàm
        
    Returns:
        comparison: Dictionary chứa kết quả so sánh
    """
    comparison = {}
    
    # Benchmark manual implementation
    start_time = time.time()
    try:
        result_manual = func_manual(image, *args, **kwargs)
        time_manual = time.time() - start_time
        error_manual = None
    except Exception as e:
        result_manual = None
        time_manual = float('inf')
        error_manual = str(e)
    
    # Benchmark library implementation
    start_time = time.time()
    try:
        result_library = func_library(image, *args, **kwargs)
        time_library = time.time() - start_time
        error_library = None
    except Exception as e:
        result_library = None
        time_library = float('inf')
        error_library = str(e)
    
    comparison['manual'] = {
        'result': result_manual,
        'time': time_manual,
        'error': error_manual
    }
    
    comparison['library'] = {
        'result': result_library,
        'time': time_library,
        'error': error_library
    }
    
    # So sánh kết quả nếu cả hai đều thành công
    if result_manual is not None and result_library is not None:
        # Ensure same shape for comparison
        if result_manual.shape != result_library.shape:
            min_h = min(result_manual.shape[0], result_library.shape[0])
            min_w = min(result_manual.shape[1], result_library.shape[1])
            result_manual_crop = result_manual[:min_h, :min_w]
            result_library_crop = result_library[:min_h, :min_w]
        else:
            result_manual_crop = result_manual
            result_library_crop = result_library
        
        # Calculate metrics
        mse = np.mean((result_manual_crop.astype(float) - result_library_crop.astype(float)) ** 2)
        max_diff = np.max(np.abs(result_manual_crop.astype(int) - result_library_crop.astype(int)))
        correlation = np.corrcoef(result_manual_crop.flatten(), result_library_crop.flatten())[0, 1]
        
        # PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        comparison['metrics'] = {
            'mse': float(mse),
            'max_difference': int(max_diff),
            'correlation': float(correlation),
            'psnr': float(psnr),
            'speedup': time_manual / time_library if time_library > 0 else float('inf')
        }
    else:
        comparison['metrics'] = None
    
    return comparison


def benchmark_batch_processing(images, processing_func, *args, **kwargs):
    """
    Benchmark xử lý batch ảnh
    
    Args:
        images: List các ảnh
        processing_func: Hàm xử lý
        *args, **kwargs: Arguments cho hàm
        
    Returns:
        benchmark: Dictionary chứa kết quả benchmark
    """
    n_images = len(images)
    times = []
    results = []
    errors = []
    
    total_start = time.time()
    
    for i, image in enumerate(images):
        start_time = time.time()
        try:
            result = processing_func(image, *args, **kwargs)
            process_time = time.time() - start_time
            
            times.append(process_time)
            results.append(result)
            errors.append(None)
            
        except Exception as e:
            process_time = time.time() - start_time
            times.append(process_time)
            results.append(None)
            errors.append(str(e))
    
    total_time = time.time() - total_start
    
    # Tính thống kê
    valid_times = [t for t, e in zip(times, errors) if e is None]
    
    benchmark = {
        'n_images': n_images,
        'total_time': total_time,
        'individual_times': times,
        'results': results,
        'errors': errors,
        'success_rate': len(valid_times) / n_images,
        'avg_time_per_image': np.mean(valid_times) if valid_times else float('inf'),
        'std_time': np.std(valid_times) if valid_times else 0,
        'min_time': np.min(valid_times) if valid_times else float('inf'),
        'max_time': np.max(valid_times) if valid_times else 0,
        'throughput': len(valid_times) / total_time if total_time > 0 else 0
    }
    
    return benchmark


def analyze_image_quality_metrics(original, processed):
    """
    Phân tích quality metrics của ảnh
    
    Args:
        original: Ảnh gốc
        processed: Ảnh đã xử lý
        
    Returns:
        metrics: Dictionary chứa các metrics
    """
    if original.shape != processed.shape:
        raise ValueError("Images must have same shape")
    
    # MSE
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # SSIM (simplified version)
    def ssim_simple(img1, img2):
        mu1 = np.mean(img1.astype(float))
        mu2 = np.mean(img2.astype(float))
        sigma1_sq = np.var(img1.astype(float))
        sigma2_sq = np.var(img2.astype(float))
        sigma12 = np.mean((img1.astype(float) - mu1) * (img2.astype(float) - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim
    
    ssim = ssim_simple(original, processed)
    
    # Correlation
    correlation = np.corrcoef(original.flatten(), processed.flatten())[0, 1]
    
    # Histogram similarity (Chi-square distance)
    hist1, _ = np.histogram(original.flatten(), bins=256, range=(0, 256))
    hist2, _ = np.histogram(processed.flatten(), bins=256, range=(0, 256))
    
    # Normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Chi-square distance
    chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
    
    # Edge preservation (using Sobel)
    def compute_edges(img):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        from core.convolution import convolution_2d_manual
        grad_x = convolution_2d_manual(img, sobel_x.astype(np.float32), padding=1)
        grad_y = convolution_2d_manual(img, sobel_y.astype(np.float32), padding=1)
        
        magnitude = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
        return magnitude
    
    edges_original = compute_edges(original)
    edges_processed = compute_edges(processed)
    
    edge_correlation = np.corrcoef(edges_original.flatten(), edges_processed.flatten())[0, 1]
    
    metrics = {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim),
        'correlation': float(correlation),
        'chi_square_distance': float(chi_square),
        'edge_preservation': float(edge_correlation)
    }
    
    return metrics


def compare_multiple_methods(image, methods_dict):
    """
    So sánh nhiều phương pháp xử lý
    
    Args:
        image: Input image
        methods_dict: Dictionary {method_name: (func, args, kwargs)}
        
    Returns:
        comparison: Dictionary chứa kết quả so sánh
    """
    comparison = {}
    
    for method_name, (func, args, kwargs) in methods_dict.items():
        start_time = time.time()
        try:
            result = func(image, *args, **kwargs)
            process_time = time.time() - start_time
            error = None
        except Exception as e:
            result = None
            process_time = float('inf')
            error = str(e)
        
        comparison[method_name] = {
            'result': result,
            'time': process_time,
            'error': error
        }
        
        # Tính quality metrics nếu có kết quả
        if result is not None:
            try:
                quality_metrics = analyze_image_quality_metrics(image, result)
                comparison[method_name]['quality_metrics'] = quality_metrics
            except:
                comparison[method_name]['quality_metrics'] = None
    
    # Tính relative performance
    valid_times = {name: data['time'] for name, data in comparison.items() 
                   if data['error'] is None and data['time'] != float('inf')}
    
    if valid_times:
        fastest_time = min(valid_times.values())
        for method_name in comparison:
            if comparison[method_name]['error'] is None:
                comparison[method_name]['relative_speed'] = \
                    comparison[method_name]['time'] / fastest_time
            else:
                comparison[method_name]['relative_speed'] = float('inf')
    
    return comparison


def generate_comparison_report(comparisons, output_file=None):
    """
    Tạo báo cáo so sánh chi tiết
    
    Args:
        comparisons: Dictionary chứa kết quả so sánh
        output_file: File output (optional)
        
    Returns:
        report: String chứa báo cáo
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPARISON REPORT")
    report_lines.append("=" * 80)
    
    for comparison_name, data in comparisons.items():
        report_lines.append(f"\n{comparison_name.upper()}")
        report_lines.append("-" * 60)
        
        if isinstance(data, dict) and 'manual' in data and 'library' in data:
            # Manual vs Library comparison
            manual_data = data['manual']
            library_data = data['library']
            
            report_lines.append("Manual Implementation:")
            report_lines.append(f"  Time: {manual_data['time']:.4f}s")
            report_lines.append(f"  Error: {manual_data['error']}")
            
            report_lines.append("Library Implementation:")
            report_lines.append(f"  Time: {library_data['time']:.4f}s")
            report_lines.append(f"  Error: {library_data['error']}")
            
            if data.get('metrics'):
                metrics = data['metrics']
                report_lines.append("Comparison Metrics:")
                report_lines.append(f"  MSE: {metrics['mse']:.6f}")
                report_lines.append(f"  Max Difference: {metrics['max_difference']}")
                report_lines.append(f"  Correlation: {metrics['correlation']:.6f}")
                report_lines.append(f"  PSNR: {metrics['psnr']:.2f} dB")
                report_lines.append(f"  Speedup: {metrics['speedup']:.2f}x")
        
        elif isinstance(data, dict):
            # Multiple methods comparison
            for method_name, method_data in data.items():
                report_lines.append(f"{method_name}:")
                report_lines.append(f"  Time: {method_data['time']:.4f}s")
                report_lines.append(f"  Error: {method_data['error']}")
                
                if method_data.get('quality_metrics'):
                    qm = method_data['quality_metrics']
                    report_lines.append(f"  PSNR: {qm['psnr']:.2f} dB")
                    report_lines.append(f"  SSIM: {qm['ssim']:.4f}")
                    report_lines.append(f"  Correlation: {qm['correlation']:.4f}")
                
                if method_data.get('relative_speed'):
                    report_lines.append(f"  Relative Speed: {method_data['relative_speed']:.2f}x")
    
    report_lines.append("\n" + "=" * 80)
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report


def statistical_significance_test(results1, results2, alpha=0.05):
    """
    Test statistical significance giữa 2 phương pháp
    
    Args:
        results1, results2: Arrays kết quả từ 2 phương pháp
        alpha: Significance level
        
    Returns:
        test_result: Dictionary chứa kết quả test
    """
    from scipy import stats
    
    # Paired t-test
    try:
        t_stat, p_value = stats.ttest_rel(results1, results2)
        
        test_result = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'alpha': alpha,
            'method': 'paired_t_test'
        }
    except Exception as e:
        test_result = {
            'error': str(e),
            'method': 'paired_t_test'
        }
    
    return test_result
