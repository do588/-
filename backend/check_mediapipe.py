import mediapipe as mp
import os
from pathlib import Path

def check_mediapipe_basic():
    """检查MediaPipe基础信息和核心模块"""
    print("=" * 60)
    print("MediaPipe 基础信息检查")
    print("=" * 60)
    
    # 版本信息
    try:
        print(f"\n📌 MediaPipe 版本: {mp.__version__}")
    except AttributeError:
        print("\n❌ 无法获取MediaPipe版本信息")
    
    # 顶层模块列表
    try:
        print(f"\n📌 MediaPipe 顶层模块:")
        modules = dir(mp)
        # 格式化输出，每行显示5个模块
        for i in range(0, len(modules), 5):
            print(f"   {' | '.join(modules[i:i+5])}")
    except Exception as e:
        print(f"\n❌ 获取顶层模块失败: {e}")
    
    # 检查tasks模块
    print("\n" + "-" * 50)
    print("📌 Tasks 模块检查")
    print("-" * 50)
    if hasattr(mp, 'tasks'):
        try:
            print(f"\n✅ Tasks 模块存在")
            print(f"\nTasks 子模块列表:")
            task_modules = dir(mp.tasks)
            for i in range(0, len(task_modules), 5):
                print(f"   {' | '.join(task_modules[i:i+5])}")
            
            # 检查tasks目录
            try:
                tasks_path = Path(mp.tasks.__path__[0])
                print(f"\n📁 Tasks 模块路径: {tasks_path.absolute()}")
                
                # 列出目录下的文件
                if tasks_path.exists():
                    print(f"\n📂 Tasks 目录下的文件/文件夹:")
                    for item in sorted(tasks_path.iterdir()):
                        item_type = "📂" if item.is_dir() else "📄"
                        print(f"   {item_type} {item.name}")
                else:
                    print(f"\n❌ Tasks 目录不存在: {tasks_path}")
            except Exception as e:
                print(f"\n❌ 获取Tasks目录信息失败: {e}")
        except Exception as e:
            print(f"\n❌ 读取Tasks模块信息失败: {e}")
    else:
        print("\n❌ Tasks 模块不存在")
    
    # 检查solutions模块
    print("\n" + "-" * 50)
    print("📌 Solutions 模块检查")
    print("-" * 50)
    if hasattr(mp, 'solutions'):
        try:
            print(f"\n✅ Solutions 模块存在")
            print(f"\nSolutions 子模块列表:")
            sol_modules = dir(mp.solutions)
            for i in range(0, len(sol_modules), 5):
                print(f"   {' | '.join(sol_modules[i:i+5])}")
        except Exception as e:
            print(f"\n❌ 读取Solutions模块信息失败: {e}")
    else:
        print("\n❌ Solutions 模块不存在")

if __name__ == "__main__":
    check_mediapipe_basic()
    print("\n" + "=" * 60)
    print("检查完成！")
    print("=" * 60)