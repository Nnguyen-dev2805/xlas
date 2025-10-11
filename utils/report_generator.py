"""
Report Generator
===============

Tạo báo cáo chi tiết cho đồ án
- HTML reports
- PDF exports
- Summary statistics
- Comparison tables

Author: Image Processing Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os


def generate_html_report(results, config, output_path="report.html"):
    """
    Tạo báo cáo HTML chi tiết
    
    Args:
        results: Kết quả processing
        config: Configuration đã sử dụng
        output_path: Đường dẫn file output
        
    Returns:
        html_content: Nội dung HTML
    """
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Processing Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                      background: #e8f4f8; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Image Processing Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📋 Dataset Summary</h2>
            <div class="metric">Total Images: {results.get('total_images', 0)}</div>
            <div class="metric">Bài 1 Success: {results['summary']['bai1_success_rate']:.1%}</div>
            <div class="metric">Avg Time: {results['summary']['bai1_avg_time']:.3f}s</div>
        </div>
        
        <div class="section">
            <h2>⚙️ Configuration</h2>
            <ul>
                <li>Bài 1 Manual: {config.get('bai1_manual', 'N/A')}</li>
                <li>Bài 2 Manual: {config.get('bai2_manual', 'N/A')}</li>
                <li>Parallel Processing: {config.get('parallel', 'N/A')}</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📊 Results Summary</h2>
            <p>Detailed results and analysis...</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return html_template


def generate_summary_statistics(results):
    """
    Tạo thống kê tổng hợp
    
    Args:
        results: Kết quả processing
        
    Returns:
        stats: Dictionary thống kê
    """
    stats = {
        'dataset_info': {
            'total_images': results.get('total_images', 0),
            'processing_time': datetime.now().isoformat()
        },
        'bai1_stats': {
            'success_rate': results['summary']['bai1_success_rate'],
            'avg_time': results['summary']['bai1_avg_time']
        },
        'bai2_stats': results['summary'].get('bai2_success_rates', {})
    }
    
    return stats


def create_comparison_table(results):
    """
    Tạo bảng so sánh
    
    Args:
        results: Kết quả processing
        
    Returns:
        df: DataFrame so sánh
    """
    data = []
    
    # Bài 1 data
    data.append({
        'Method': 'Bài 1 (Histogram)',
        'Success Rate': results['summary']['bai1_success_rate'],
        'Avg Time (s)': results['summary']['bai1_avg_time']
    })
    
    # Bài 2 data
    for kernel, success_rate in results['summary'].get('bai2_success_rates', {}).items():
        avg_time = results['summary'].get('bai2_avg_times', {}).get(kernel, 0)
        data.append({
            'Method': f'Bài 2 ({kernel.title()})',
            'Success Rate': success_rate,
            'Avg Time (s)': avg_time
        })
    
    df = pd.DataFrame(data)
    return df


def export_results_json(results, output_path="results.json"):
    """
    Export kết quả ra JSON
    
    Args:
        results: Kết quả processing
        output_path: Đường dẫn file output
    """
    # Convert numpy arrays to lists for JSON serialization
    export_data = {
        'summary': results['summary'],
        'total_images': results.get('total_images', 0),
        'export_time': datetime.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


def create_performance_chart(results, output_path="performance_chart.png"):
    """
    Tạo biểu đồ performance
    
    Args:
        results: Kết quả processing
        output_path: Đường dẫn file output
    """
    try:
        # Prepare data
        methods = ['Bài 1']
        times = [results['summary']['bai1_avg_time']]
        
        # Add Bài 2 data if available
        for kernel, avg_time in results['summary'].get('bai2_avg_times', {}).items():
            methods.append(f'Bài 2 ({kernel})')
            times.append(avg_time)
        
        # Create chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, times, alpha=0.7)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.title('Average Processing Time by Method')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not create performance chart: {e}")


def generate_markdown_report(results, config, output_path="report.md"):
    """
    Tạo báo cáo Markdown
    
    Args:
        results: Kết quả processing
        config: Configuration
        output_path: Đường dẫn file output
        
    Returns:
        markdown_content: Nội dung markdown
    """
    
    markdown_content = f"""# 📊 Image Processing Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 Dataset Summary

- **Total Images:** {results.get('total_images', 0)}
- **Processing Date:** {datetime.now().strftime('%Y-%m-%d')}

## ⚙️ Configuration

- **Bài 1 Method:** {'Manual' if config.get('bai1_manual') else 'Library'}
- **Bài 2 Method:** {'Manual' if config.get('bai2_manual') else 'Library'}
- **Parallel Processing:** {config.get('parallel', False)}

## 📈 Bài 1 Results (Histogram Processing)

- **Success Rate:** {results['summary']['bai1_success_rate']:.1%}
- **Average Time:** {results['summary']['bai1_avg_time']:.3f} seconds per image

## 🔧 Bài 2 Results (Convolution & Filtering)

"""
    
    # Add Bài 2 results if available
    for kernel, success_rate in results['summary'].get('bai2_success_rates', {}).items():
        avg_time = results['summary'].get('bai2_avg_times', {}).get(kernel, 0)
        markdown_content += f"### {kernel.title()} Kernel\n"
        markdown_content += f"- **Success Rate:** {success_rate:.1%}\n"
        markdown_content += f"- **Average Time:** {avg_time:.3f} seconds per image\n\n"
    
    markdown_content += """## 📊 Summary

This report shows the performance comparison between manual and library implementations for image processing tasks.

---
*Generated by Image Processing Report System*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return markdown_content


def create_full_report_package(results, config, output_dir="report_package"):
    """
    Tạo package báo cáo đầy đủ
    
    Args:
        results: Kết quả processing
        config: Configuration
        output_dir: Thư mục output
        
    Returns:
        created_files: List các file đã tạo
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    created_files = []
    
    try:
        # HTML Report
        html_path = os.path.join(output_dir, "report.html")
        generate_html_report(results, config, html_path)
        created_files.append(html_path)
        
        # Markdown Report
        md_path = os.path.join(output_dir, "report.md")
        generate_markdown_report(results, config, md_path)
        created_files.append(md_path)
        
        # JSON Export
        json_path = os.path.join(output_dir, "results.json")
        export_results_json(results, json_path)
        created_files.append(json_path)
        
        # Performance Chart
        chart_path = os.path.join(output_dir, "performance_chart.png")
        create_performance_chart(results, chart_path)
        created_files.append(chart_path)
        
        # Summary Statistics
        stats = generate_summary_statistics(results)
        stats_path = os.path.join(output_dir, "summary_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        created_files.append(stats_path)
        
        # Comparison Table
        df = create_comparison_table(results)
        table_path = os.path.join(output_dir, "comparison_table.csv")
        df.to_csv(table_path, index=False)
        created_files.append(table_path)
        
    except Exception as e:
        print(f"Warning: Some report files could not be created: {e}")
    
    return created_files
