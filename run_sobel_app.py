import os
import sys
import subprocess

def main():
    print("="*70)
    print(" "*15 + "SOBEL EDGE DETECTION - Streamlit App")
    print("="*70)
    print("\nĐang khởi động Streamlit app...")
    print("App sẽ mở tự động trong browser")
    print("Nhấn Ctrl+C để tắt server\n")
    print("="*70 + "\n")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app_sobel.py")
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            app_path,
            "--server.port", "8502",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n✓ Đã tắt Streamlit server. Bye!")
    except Exception as e:
        print(f"\nLỗi: {e}")
        print("\nThử chạy trực tiếp:")
        print(f"   streamlit run {app_path}")

if __name__ == "__main__":
    main()
