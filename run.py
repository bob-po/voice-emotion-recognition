#!/usr/bin/env python3
"""
语音情绪识别系统快速启动脚本
"""

import sys
import os
import subprocess
import argparse

def check_dependencies():
    """
    检查依赖是否安装
    """
    required_packages = [
        'torch', 'transformers', 'librosa', 'soundfile', 
        'numpy', 'gradio', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def run_web_interface():
    """
    启动Web界面
    """
    print("🚀 启动Web界面...")
    try:
        subprocess.run([sys.executable, "web_interface.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Web界面已停止")
    except Exception as e:
        print(f"❌ 启动Web界面失败: {e}")

def run_cli_tool():
    """
    启动命令行工具
    """
    print("💻 启动命令行工具...")
    try:
        # 显示帮助信息
        subprocess.run([sys.executable, "cli_tool.py", "--help"], check=True)
    except Exception as e:
        print(f"❌ 启动命令行工具失败: {e}")

def run_test():
    """
    运行系统测试
    """
    print("🧪 运行系统测试...")
    try:
        subprocess.run([sys.executable, "test_system.py"], check=True)
    except Exception as e:
        print(f"❌ 运行测试失败: {e}")

def install_dependencies():
    """
    安装依赖
    """
    print("📦 安装依赖包...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ 依赖安装完成")
    except Exception as e:
        print(f"❌ 依赖安装失败: {e}")

def show_info():
    """
    显示系统信息
    """
    print("🎤 语音情绪识别系统")
    print("=" * 40)
    print("基于Hugging Face预训练模型的语音情绪识别系统")
    print("支持7种情绪分类：愤怒、厌恶、恐惧、快乐、悲伤、惊讶、中性")
    print("\n📁 项目文件:")
    
    files = [
        ("emotion_speech_recognition.py", "核心识别类"),
        ("web_interface.py", "Web界面"),
        ("cli_tool.py", "命令行工具"),
        ("test_system.py", "系统测试"),
        ("requirements.txt", "依赖文件"),
        ("README.md", "项目说明")
    ]
    
    for file, desc in files:
        if os.path.exists(file):
            print(f"   ✅ {file} - {desc}")
        else:
            print(f"   ❌ {file} - {desc} (缺失)")
    
    print("\n🚀 使用方法:")
    print("   python run.py web          # 启动Web界面")
    print("   python run.py cli          # 启动命令行工具")
    print("   python run.py test         # 运行系统测试")
    print("   python run.py install      # 安装依赖")
    print("   python run.py info         # 显示系统信息")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="语音情绪识别系统快速启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py web          # 启动Web界面
  python run.py cli          # 启动命令行工具
  python run.py test         # 运行系统测试
  python run.py install      # 安装依赖
  python run.py info         # 显示系统信息
        """
    )
    
    parser.add_argument(
        "command",
        choices=["web", "cli", "test", "install", "info"],
        help="要执行的命令"
    )
    
    args = parser.parse_args()
    
    # 检查依赖（除了install命令）
    if args.command != "install" and not check_dependencies():
        print("\n请先运行 'python run.py install' 安装依赖")
        return
    
    # 执行命令
    if args.command == "web":
        run_web_interface()
    elif args.command == "cli":
        run_cli_tool()
    elif args.command == "test":
        run_test()
    elif args.command == "install":
        install_dependencies()
    elif args.command == "info":
        show_info()

if __name__ == "__main__":
    main() 