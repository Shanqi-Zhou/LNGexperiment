#!/usr/bin/env python3
"""
LNG模型评估脚本
基于技术路线的标准实现
"""

import sys
import logging
import argparse
from pathlib import Path

# 添加src路径到系统路径
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from src.eval import LNGEvaluator, EvaluationConfig


def setup_logging(level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LNG模型评估脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本配置
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/eval.yaml',
        help='评估配置文件路径'
    )
    
    # 模型路径
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='训练好的模型文件路径'
    )
    
    parser.add_argument(
        '--compare-models',
        nargs='+',
        help='多个模型路径用于比较'
    )
    
    # 数据路径
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/sim_lng',
        help='测试数据目录'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='评估结果输出目录'
    )
    
    # 模型参数
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['transformer', 'mlr', 'gpr', 'baseline'],
        help='模型类型'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='评估批次大小'
    )
    
    # 评估选项
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['mse', 'rmse', 'mae', 'r2', 'mape'],
        help='要计算的指标'
    )
    
    parser.add_argument(
        '--create-plots',
        action='store_true',
        default=True,
        help='创建评估图表'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true', 
        default=True,
        help='保存模型预测结果'
    )
    
    # 设备设置
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='计算设备'
    )
    
    # 日志设置
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def main():
    """主评估函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    logger = setup_logging(log_level)
    
    logger.info("启动LNG模型评估...")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    try:
        # 创建评估配置
        eval_config = EvaluationConfig(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_type=args.model_type,
            batch_size=args.batch_size,
            metrics=args.metrics,
            create_plots=args.create_plots,
            save_predictions=args.save_predictions,
            device=args.device,
            log_level=args.log_level
        )
        
        # 创建评估器
        evaluator = LNGEvaluator(config=eval_config, logger=logger)
        
        # 执行评估
        if args.compare_models:
            # 多模型比较
            logger.info(f"比较{len(args.compare_models)}个模型...")
            results = evaluator.compare_models(args.compare_models)
        else:
            # 单模型评估
            logger.info("执行单模型评估...")
            results = evaluator.evaluate_single_model()
        
        # 打印结果摘要
        logger.info("=" * 60)
        logger.info("评估结果摘要:")
        logger.info("=" * 60)
        
        if 'metrics' in results:
            metrics = results['metrics']
            if 'r2_score' in metrics:
                logger.info(f"R² Score: {metrics['r2_score']:.6f}")
            if 'rmse' in metrics:
                logger.info(f"RMSE: {metrics['rmse']:.6f}")
            if 'mae' in metrics:
                logger.info(f"MAE: {metrics['mae']:.6f}")
            if 'nmbe_percent' in metrics:
                logger.info(f"NMBE: {metrics['nmbe_percent']:.3f}%")
            if 'cv_rmse_percent' in metrics:
                logger.info(f"CV(RMSE): {metrics['cv_rmse_percent']:.3f}%")
        
        logger.info(f"结果保存到: {args.output_dir}")
        logger.info("评估完成!")
        
        return 0
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())