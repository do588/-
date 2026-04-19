import mediapipe as mp

def check_mediapipe_detailed():
    """详细检查MediaPipe的API结构，包括嵌套子模块"""
    print("=" * 70)
    print("MediaPipe API 详细结构检查")
    print("=" * 70)
    
    # 基础版本信息
    try:
        print(f"\n📌 MediaPipe 版本: {mp.__version__}")
    except AttributeError:
        print(f"\n❌ 无法获取版本信息")
    
    # 顶层模块
    print(f"\n📌 MediaPipe 顶层模块列表:")
    try:
        top_modules = dir(mp)
        # 过滤掉私有模块（以_开头的）
        public_modules = [m for m in top_modules if not m.startswith('_')]
        for i in range(0, len(public_modules), 4):
            print(f"   {' | '.join(public_modules[i:i+4])}")
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
    
    # 详细检查tasks模块层级
    print("\n" + "-" * 60)
    print("📌 Tasks 模块层级详细检查")
    print("-" * 60)
    
    # 检查tasks根模块
    if hasattr(mp, 'tasks'):
        print("\n✅ mp.tasks 存在")
        
        # 检查python子模块
        try:
            import mediapipe.tasks.python as tasks_python
            print("\n✅ mp.tasks.python 存在")
            print(f"\n📌 tasks.python 子模块:")
            py_modules = [m for m in dir(tasks_python) if not m.startswith('_')]
            for i in range(0, len(py_modules), 4):
                print(f"   {' | '.join(py_modules[i:i+4])}")
            
            # 检查vision子模块
            try:
                import mediapipe.tasks.python.vision as tasks_vision
                print("\n✅ mp.tasks.python.vision 存在")
                print(f"\n📌 tasks.python.vision 子模块:")
                vision_modules = [m for m in dir(tasks_vision) if not m.startswith('_')]
                for i in range(0, len(vision_modules), 3):
                    print(f"   {' | '.join(vision_modules[i:i+3])}")
                
                # 检查vision中的关键类/函数
                key_vision_items = ['ImageClassifier', 'ObjectDetector', 'PoseLandmarker', 'HandLandmarker']
                print(f"\n🔍 vision模块关键组件检查:")
                for item in key_vision_items:
                    if hasattr(tasks_vision, item):
                        print(f"   ✅ {item} 存在")
                    else:
                        print(f"   ❌ {item} 不存在")
                        
            except ImportError as e:
                print(f"\n❌ 无法导入 mp.tasks.python.vision: {e}")
            except Exception as e:
                print(f"\n❌ 检查vision模块失败: {e}")
            
            # 检查core子模块
            try:
                import mediapipe.tasks.python.core as tasks_core
                print("\n✅ mp.tasks.python.core 存在")
                print(f"\n📌 tasks.python.core 子模块:")
                core_modules = [m for m in dir(tasks_core) if not m.startswith('_')]
                for i in range(0, len(core_modules), 4):
                    print(f"   {' | '.join(core_modules[i:i+4])}")
                    
                # 检查core中的关键组件
                key_core_items = ['BaseOptions', 'RunningMode', 'TaskInfo']
                print(f"\n🔍 core模块关键组件检查:")
                for item in key_core_items:
                    if hasattr(tasks_core, item):
                        print(f"   ✅ {item} 存在")
                    else:
                        print(f"   ❌ {item} 不存在")
                        
            except ImportError as e:
                print(f"\n❌ 无法导入 mp.tasks.python.core: {e}")
            except Exception as e:
                print(f"\n❌ 检查core模块失败: {e}")
                
        except ImportError as e:
            print(f"\n❌ 无法导入 mp.tasks.python: {e}")
        except Exception as e:
            print(f"\n❌ 检查tasks.python模块失败: {e}")
    else:
        print("\n❌ mp.tasks 模块不存在")
    
    # 检查solutions模块
    print("\n" + "-" * 60)
    print("📌 Solutions 模块检查")
    print("-" * 60)
    if hasattr(mp, 'solutions'):
        print("\n✅ mp.solutions 存在")
        print(f"\n📌 solutions 子模块:")
        try:
            sol_modules = [m for m in dir(mp.solutions) if not m.startswith('_')]
            for i in range(0, len(sol_modules), 4):
                print(f"   {' | '.join(sol_modules[i:i+4])}")
            
            # 检查常用的solutions组件
            key_solutions = ['hands', 'pose', 'face_mesh', 'holistic', 'drawing_utils']
            print(f"\n🔍 solutions常用组件检查:")
            for item in key_solutions:
                if hasattr(mp.solutions, item):
                    print(f"   ✅ {item} 存在")
                else:
                    print(f"   ❌ {item} 不存在")
                    
        except Exception as e:
            print(f"\n❌ 检查solutions模块失败: {e}")
    else:
        print("\n❌ mp.solutions 模块不存在")

if __name__ == "__main__":
    check_mediapipe_detailed()
    print("\n" + "=" * 70)
    print("详细检查完成！")
    print("=" * 70)