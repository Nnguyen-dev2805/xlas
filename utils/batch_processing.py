"""
Batch Processing Tools
=====================

Công cụ xử lý batch nhiều ảnh
- Batch processing pipeline
- Progress tracking
- Error handling
- Results aggregation

Author: Image Processing Team
"""

import numpy as np
import os
from typing import List, Dict, Callable, Any, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class BatchProcessor:
    """
    Class xử lý batch ảnh với progress tracking
    """
    
    def __init__(self, max_workers=None):
        """
        Args:
            max_workers: Số thread tối đa cho parallel processing
        """
        self.max_workers = max_workers
        self.results = []
        self.errors = []
        self.processing_times = []
    
    def process_single_image(self, image, image_id, processing_pipeline):
        """
        Xử lý một ảnh qua pipeline
        
        Args:
            image: Input image
            image_id: ID của ảnh
            processing_pipeline: List các (function, args, kwargs)
            
        Returns:
            result: Dictionary chứa kết quả
        """
        start_time = time.time()
        
        try:
            current_image = image.copy()
            pipeline_results = {'original': image}
            
            for i, (func, args, kwargs) in enumerate(processing_pipeline):
                step_start = time.time()
                
                # Apply processing function
                if args and kwargs:
                    result = func(current_image, *args, **kwargs)
                elif args:
                    result = func(current_image, *args)
                elif kwargs:
                    result = func(current_image, **kwargs)
                else:
                    result = func(current_image)
                
                step_time = time.time() - step_start
                
                # Store result
                step_name = f"step_{i+1}_{func.__name__}"
                pipeline_results[step_name] = result
                pipeline_results[f"{step_name}_time"] = step_time
                
                # Update current image for next step
                current_image = result
            
            total_time = time.time() - start_time
            
            return {
                'image_id': image_id,
                'success': True,
                'results': pipeline_results,
                'total_time': total_time,
                'error': None
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            
            return {
                'image_id': image_id,
                'success': False,
                'results': None,
                'total_time': total_time,
                'error': str(e)
            }
    
    def process_batch(self, images, processing_pipeline, image_ids=None, 
                     parallel=False, progress_callback=None):
        """
        Xử lý batch ảnh
        
        Args:
            images: List các ảnh
            processing_pipeline: Pipeline xử lý
            image_ids: List ID của ảnh
            parallel: Có xử lý parallel không
            progress_callback: Callback function cho progress
            
        Returns:
            batch_results: Dictionary chứa kết quả batch
        """
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]
        
        if len(images) != len(image_ids):
            raise ValueError("Number of images and image_ids must match")
        
        batch_start_time = time.time()
        results = []
        
        if parallel and self.max_workers != 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_id = {
                    executor.submit(self.process_single_image, img, img_id, processing_pipeline): img_id
                    for img, img_id in zip(images, image_ids)
                }
                
                # Collect results
                completed = 0
                for future in as_completed(future_to_id):
                    result = future.result()
                    results.append(result)
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(images))
        
        else:
            # Sequential processing
            for i, (image, image_id) in enumerate(zip(images, image_ids)):
                result = self.process_single_image(image, image_id, processing_pipeline)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(images))
        
        batch_total_time = time.time() - batch_start_time
        
        # Aggregate results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        batch_results = {
            'total_images': len(images),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(images),
            'batch_time': batch_total_time,
            'avg_time_per_image': np.mean([r['total_time'] for r in successful_results]) if successful_results else 0,
            'results': results,
            'successful_results': successful_results,
            'failed_results': failed_results
        }
        
        return batch_results


def create_processing_pipeline_bai1(use_manual=True):
    """
    Tạo pipeline cho Bài 1
    
    Args:
        use_manual: Sử dụng manual implementation
        
    Returns:
        pipeline: List các processing steps
    """
    if use_manual:
        from core.image_ops import rgb_to_grayscale_manual
        from core.histogram import (calculate_histogram_manual, 
                                    histogram_equalization_manual,
                                    histogram_narrowing_manual)
        
        pipeline = [
            (rgb_to_grayscale_manual, [], {}),
            (histogram_equalization_manual, [], {}),
            (histogram_narrowing_manual, [], {'min_val': 30, 'max_val': 80})
        ]
    else:
        from core.image_ops import rgb_to_grayscale_library
        from core.histogram import (calculate_histogram_library,
                                    histogram_equalization_library)
        
        pipeline = [
            (rgb_to_grayscale_library, [], {}),
            (histogram_equalization_library, [], {})
        ]
    
    return pipeline


def create_processing_pipeline_bai2(use_manual=True, kernel_type='gaussian'):
    """
    Tạo pipeline cho Bài 2
    
    Args:
        use_manual: Sử dụng manual implementation
        kernel_type: Loại kernel ('gaussian', 'sobel', 'sharpen')
        
    Returns:
        pipeline: List các processing steps
    """
    from core.image_ops import rgb_to_grayscale_manual, rgb_to_grayscale_library
    
    if use_manual:
        from core.convolution import convolution_2d_manual, median_filter_manual
        from filters.kernels import create_gaussian_kernel, get_sobel_kernels, get_sharpen_kernel
        
        # Chọn kernel
        if kernel_type == 'gaussian':
            kernel = create_gaussian_kernel(5, 1.2)
        elif kernel_type == 'sobel':
            kernel, _ = get_sobel_kernels()
        elif kernel_type == 'sharpen':
            kernel = get_sharpen_kernel()
        else:
            kernel = create_gaussian_kernel(3, 0.8)
        
        pipeline = [
            (rgb_to_grayscale_manual, [], {}),
            (convolution_2d_manual, [kernel], {'padding': 1, 'stride': 1}),
            (median_filter_manual, [3], {})
        ]
    else:
        from filters.blur_filters import gaussian_blur_library
        from core.convolution import median_filter_library
        
        pipeline = [
            (rgb_to_grayscale_library, [], {}),
            (gaussian_blur_library, [], {'kernel_size': 5, 'sigma': 1.2}),
            (median_filter_library, [3], {})
        ]
    
    return pipeline


def process_image_dataset(images, dataset_name="dataset", output_dir=None,
                         bai1_manual=True, bai2_manual=True, 
                         save_intermediate=False, parallel=True):
    """
    Xử lý toàn bộ dataset ảnh cho cả Bài 1 và Bài 2
    
    Args:
        images: List các ảnh
        dataset_name: Tên dataset
        output_dir: Thư mục output
        bai1_manual: Sử dụng manual cho Bài 1
        bai2_manual: Sử dụng manual cho Bài 2
        save_intermediate: Lưu kết quả trung gian
        parallel: Xử lý parallel
        
    Returns:
        dataset_results: Dictionary chứa tất cả kết quả
    """
    print(f"🚀 Processing dataset: {dataset_name}")
    print(f"📊 Total images: {len(images)}")
    print(f"🔧 Bài 1 Manual: {bai1_manual}, Bài 2 Manual: {bai2_manual}")
    
    # Tạo processors
    processor = BatchProcessor(max_workers=4 if parallel else 1)
    
    # Progress callback
    def progress_callback(completed, total):
        progress = (completed / total) * 100
        print(f"  Progress: {completed}/{total} ({progress:.1f}%)")
    
    # Process Bài 1
    print("\n📈 Processing Bài 1 (Histogram Processing)...")
    bai1_pipeline = create_processing_pipeline_bai1(use_manual=bai1_manual)
    bai1_results = processor.process_batch(
        images, bai1_pipeline, 
        image_ids=[f"{dataset_name}_img_{i}" for i in range(len(images))],
        parallel=parallel,
        progress_callback=progress_callback
    )
    
    # Process Bài 2 với different kernels
    bai2_results = {}
    kernel_types = ['gaussian', 'sobel', 'sharpen']
    
    for kernel_type in kernel_types:
        print(f"\n🔧 Processing Bài 2 ({kernel_type.title()} Kernel)...")
        bai2_pipeline = create_processing_pipeline_bai2(
            use_manual=bai2_manual, 
            kernel_type=kernel_type
        )
        
        results = processor.process_batch(
            images, bai2_pipeline,
            image_ids=[f"{dataset_name}_img_{i}_{kernel_type}" for i in range(len(images))],
            parallel=parallel,
            progress_callback=progress_callback
        )
        
        bai2_results[kernel_type] = results
    
    # Aggregate all results
    dataset_results = {
        'dataset_name': dataset_name,
        'total_images': len(images),
        'processing_config': {
            'bai1_manual': bai1_manual,
            'bai2_manual': bai2_manual,
            'parallel': parallel
        },
        'bai1_results': bai1_results,
        'bai2_results': bai2_results,
        'summary': {
            'bai1_success_rate': bai1_results['success_rate'],
            'bai1_avg_time': bai1_results['avg_time_per_image'],
            'bai2_success_rates': {k: v['success_rate'] for k, v in bai2_results.items()},
            'bai2_avg_times': {k: v['avg_time_per_image'] for k, v in bai2_results.items()}
        }
    }
    
    # Save results if output_dir specified
    if output_dir:
        save_batch_results(dataset_results, output_dir, save_intermediate)
    
    print(f"\n✅ Dataset processing completed!")
    print(f"📊 Bài 1 Success Rate: {bai1_results['success_rate']:.2%}")
    for kernel_type, results in bai2_results.items():
        print(f"📊 Bài 2 ({kernel_type}) Success Rate: {results['success_rate']:.2%}")
    
    return dataset_results


def save_batch_results(dataset_results, output_dir, save_intermediate=False):
    """
    Lưu kết quả batch processing
    
    Args:
        dataset_results: Kết quả dataset
        output_dir: Thư mục output
        save_intermediate: Lưu kết quả trung gian
    """
    import json
    import pickle
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_name = dataset_results['dataset_name']
    
    # Save summary as JSON
    summary_file = os.path.join(output_dir, f"{dataset_name}_summary.json")
    with open(summary_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        summary_data = {
            'dataset_name': dataset_results['dataset_name'],
            'total_images': dataset_results['total_images'],
            'processing_config': dataset_results['processing_config'],
            'summary': dataset_results['summary']
        }
        json.dump(summary_data, f, indent=2)
    
    # Save full results as pickle
    results_file = os.path.join(output_dir, f"{dataset_name}_full_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(dataset_results, f)
    
    # Save individual images if requested
    if save_intermediate:
        from core.image_ops import save_image
        
        # Create subdirectories
        bai1_dir = os.path.join(output_dir, f"{dataset_name}_bai1")
        bai2_dir = os.path.join(output_dir, f"{dataset_name}_bai2")
        
        os.makedirs(bai1_dir, exist_ok=True)
        os.makedirs(bai2_dir, exist_ok=True)
        
        # Save Bài 1 results
        for result in dataset_results['bai1_results']['successful_results']:
            image_id = result['image_id']
            results = result['results']
            
            for step_name, step_result in results.items():
                if isinstance(step_result, np.ndarray) and step_name != 'original':
                    filename = f"{image_id}_{step_name}.png"
                    filepath = os.path.join(bai1_dir, filename)
                    save_image(step_result, filepath)
        
        # Save Bài 2 results
        for kernel_type, kernel_results in dataset_results['bai2_results'].items():
            kernel_dir = os.path.join(bai2_dir, kernel_type)
            os.makedirs(kernel_dir, exist_ok=True)
            
            for result in kernel_results['successful_results']:
                image_id = result['image_id']
                results = result['results']
                
                for step_name, step_result in results.items():
                    if isinstance(step_result, np.ndarray) and step_name != 'original':
                        filename = f"{image_id}_{step_name}.png"
                        filepath = os.path.join(kernel_dir, filename)
                        save_image(step_result, filepath)
    
    print(f"💾 Results saved to: {output_dir}")


def load_batch_results(results_file):
    """
    Load kết quả batch processing
    
    Args:
        results_file: File chứa kết quả
        
    Returns:
        dataset_results: Kết quả đã load
    """
    import pickle
    
    with open(results_file, 'rb') as f:
        dataset_results = pickle.load(f)
    
    return dataset_results
