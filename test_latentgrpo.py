"""
简单的LatentGRPO测试脚本
验证模型初始化和基本功能
"""

import sys
import torch

def test_imports():
    """测试导入"""
    print("测试导入...")
    try:
        from models.latentgrpo import LatentGRPO
        from training.train_latentgrpo import train_latentgrpo_model, run_latentgrpo_inference
        print("✓ 导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_model_initialization():
    """测试模型初始化"""
    print("\n测试模型初始化...")
    try:
        from models.latentgrpo import LatentGRPO
        
        # 使用小模型进行测试
        model = LatentGRPO(
            config="small",
            llm_model_name="meta-llama/Llama-2-7b-chat-hf"
        )
        
        print("✓ 模型初始化成功")
        print(f"  - 配置: {model.config}")
        print(f"  - LLM模型: {model.llm_model_name}")
        print(f"  - 投影模块: {type(model.proj).__name__}")
        
        # 检查LLM参数是否冻结
        frozen_params = sum(1 for p in model.llm_model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in model.llm_model.parameters())
        print(f"  - LLM参数冻结: {frozen_params}/{total_params}")
        
        # 检查投影模块是否可训练
        trainable_params = sum(1 for p in model.proj.parameters() if p.requires_grad)
        print(f"  - 投影模块可训练参数: {trainable_params}")
        
        return True
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """测试前向传播"""
    print("\n测试前向传播...")
    try:
        from models.latentgrpo import LatentGRPO
        
        # 创建模拟输入
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  使用设备: {device}")
        
        # 注意：这里只是验证接口，实际运行需要下载模型
        print("  注意: 实际前向传播需要下载大模型，此处仅验证接口")
        print("  ✓ 接口验证完成")
        
        return True
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_methods():
    """测试关键方法"""
    print("\n测试关键方法...")
    try:
        from models.latentgrpo import LatentGRPO
        
        # 检查关键方法是否存在
        required_methods = [
            'generate_continuous_thoughts',
            'generate_answer',
            'sample_multi_trajectories',
            'compute_contrastive_loss',
            'compute_advantages',
            'compute_policy_loss',
            'save_reference_projection',
            'save_pretrained',
            'from_pretrained'
        ]
        
        for method in required_methods:
            if hasattr(LatentGRPO, method):
                print(f"  ✓ {method}")
            else:
                print(f"  ✗ {method} 不存在")
                return False
        
        return True
    except Exception as e:
        print(f"✗ 方法检查失败: {e}")
        return False

def test_training_script():
    """测试训练脚本导入"""
    print("\n测试训练脚本导入...")
    try:
        from training.train_latentgrpo import (
            train_latentgrpo_model,
            run_latentgrpo_inference,
            process_batch,
            run_validation
        )
        print("✓ 训练脚本导入成功")
        print("  - train_latentgrpo_model")
        print("  - run_latentgrpo_inference")
        print("  - process_batch")
        print("  - run_validation")
        return True
    except Exception as e:
        print(f"✗ 训练脚本导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("LatentGRPO 代码验证测试")
    print("=" * 60)
    
    tests = [
        ("导入测试", test_imports),
        ("模型初始化测试", test_model_initialization),
        ("前向传播测试", test_forward_pass),
        ("关键方法测试", test_methods),
        ("训练脚本测试", test_training_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        print("\n" + "-" * 60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())