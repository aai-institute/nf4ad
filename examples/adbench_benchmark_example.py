"""
Example scripts for running ADBench benchmarks.
"""
from pathlib import Path
from nf4ad.adbench_benchmark import (
    ADBenchBenchmark,
    BenchmarkConfig,
    FlowConfig,
    ADBENCH_CLASSICAL_DATASETS,
)


def example_single_dataset():
    """Example: Run on a single dataset."""
    print("Example 1: Single Dataset Benchmark")
    print("="*60)
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/single"),
        device="cuda",
        verbose=True,
    )
    
    flow_config = FlowConfig(
        coupling_blocks=8,
        hidden_dim=128,
        lr=1e-3,
        batch_size=64,
        epochs=100,
        patience=10,
    )
    
    benchmark = ADBenchBenchmark(config)
    results = benchmark.run_on_datasets(
        datasets=["6_cardio"],
        flow_config=flow_config,
    )
    
    benchmark.save_results("single_dataset_example")
    print(benchmark.summary())


def example_multiple_datasets():
    """Example: Run on multiple datasets."""
    print("\nExample 2: Multiple Datasets Benchmark")
    print("="*60)
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/multi"),
        device="cuda",
        verbose=True,
    )
    
    flow_config = FlowConfig(
        coupling_blocks=8,
        hidden_dim=128,
        lr=1e-3,
        batch_size=64,
        epochs=100,
        patience=10,
    )
    
    # Select a subset of datasets
    test_datasets = [
        "6_cardio",
        "2_annthyroid",
        "38_thyroid",
        "4_breastw",
        "30_satellite",
    ]
    
    benchmark = ADBenchBenchmark(config)
    results = benchmark.run_on_datasets(
        datasets=test_datasets,
        flow_config=flow_config,
    )
    
    benchmark.save_results("multi_dataset_example")
    print("\nSummary:")
    print(benchmark.summary())


def example_hyperparameter_search():
    """Example: Hyperparameter search on selected datasets."""
    print("\nExample 3: Hyperparameter Search")
    print("="*60)
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/hypersearch"),
        device="cuda",
        verbose=False,  # Less verbose for grid search
    )
    
    # Define parameter grid
    param_grid = {
        'coupling_blocks': [4, 8, 12],
        'hidden_dim': [64, 128, 256],
        'lr': [1e-4, 1e-3, 1e-2],
        'batch_size': [32, 64],
        'epochs': [100],
        'patience': [10],
    }
    
    # Select datasets for tuning
    tune_datasets = [
        "6_cardio",
        "2_annthyroid",
        "38_thyroid",
    ]
    
    benchmark = ADBenchBenchmark(config)
    results_df = benchmark.hyperparameter_search(
        datasets=tune_datasets,
        param_grid=param_grid,
        n_trials=20,  # Limit to 20 random configurations
    )
    
    # Find best configuration
    print("\nTop 5 configurations by ROC-AUC:")
    print(results_df.nlargest(5, 'roc_auc')[
        ['dataset', 'roc_auc', 'pr_auc', 'best_f1', 
         'config_coupling_blocks', 'config_hidden_dim', 'config_lr']
    ])


def example_full_benchmark():
    """Example: Run on all ADBench classical datasets."""
    print("\nExample 4: Full Benchmark (All Datasets)")
    print("="*60)
    print(f"Total datasets: {len(ADBENCH_CLASSICAL_DATASETS)}")
    print("WARNING: This will take a long time!")
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/full"),
        device="cuda",
        verbose=True,
        n_jobs=4,  # Parallel processing
    )
    
    flow_config = FlowConfig(
        coupling_blocks=8,
        hidden_dim=128,
        lr=1e-3,
        batch_size=64,
        epochs=100,
        patience=10,
    )
    
    benchmark = ADBenchBenchmark(config)
    results = benchmark.run_on_datasets(
        datasets=ADBENCH_CLASSICAL_DATASETS,
        flow_config=flow_config,
        parallel=True,
    )
    
    benchmark.save_results("full_benchmark")
    
    # Summary statistics
    print("\nOverall Summary:")
    summary = benchmark.summary()
    print(summary)
    
    # Save summary
    summary.to_csv(config.output_dir / "summary_statistics.csv")


def example_usflow_benchmark():
    """Example: Run benchmark using USFlow model."""
    print("\nExample 5: USFlow Model Benchmark")
    print("="*60)
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/usflow"),
        device="cuda",
        verbose=True,
    )
    
    # Configure USFlow model
    flow_config = FlowConfig(
        flow_type="usflow",  # Use USFlow instead of NonUSFlow
        coupling_blocks=8,
        hidden_dim=128,
        lr=1e-3,
        batch_size=64,
        epochs=100,
        patience=10,
        lu_transform=1,
        householder=0,
        affine_conjugation=True,
        prior_scale=1.0,
        masktype="checkerboard",
    )
    
    # Select datasets to test
    test_datasets = [
        "6_cardio",
        "2_annthyroid",
        "38_thyroid",
    ]
    
    benchmark = ADBenchBenchmark(config)
    results = benchmark.run_on_datasets(
        datasets=test_datasets,
        flow_config=flow_config,
    )
    
    benchmark.save_results("usflow_benchmark")
    print("\nUSFlow Benchmark Summary:")
    print(benchmark.summary())


def example_compare_flows():
    """Example: Compare NonUSFlow vs USFlow."""
    print("\nExample 6: Compare NonUSFlow vs USFlow")
    print("="*60)
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/comparison"),
        device="cuda",
        verbose=True,
    )
    
    test_datasets = ["6_cardio", "2_annthyroid"]
    
    # Test both flow types
    benchmark = ADBenchBenchmark(config)
    
    for flow_type in ["nonusflow", "usflow"]:
        print(f"\n{'='*60}")
        print(f"Testing {flow_type.upper()}")
        print(f"{'='*60}")
        
        flow_config = FlowConfig(
            flow_type=flow_type,
            coupling_blocks=8,
            hidden_dim=128,
            lr=1e-3,
            batch_size=64,
            epochs=50,
            patience=10,
        )
        
        # Add USFlow-specific parameters
        if flow_type == "usflow":
            flow_config.lu_transform = 1
            flow_config.householder = 0
            flow_config.masktype = "checkerboard"
        
        benchmark.run_on_datasets(
            datasets=test_datasets,
            flow_config=flow_config,
        )
    
    # Compare results
    df = benchmark.results_to_dataframe()
    comparison = df.groupby(['dataset', 'config_flow_type']).agg({
        'roc_auc': 'mean',
        'pr_auc': 'mean',
        'best_f1': 'mean',
        'training_time': 'mean',
    }).round(4)
    
    print("\nComparison Results:")
    print(comparison)
    
    benchmark.save_results("flow_comparison")


def example_usflow_hyperparameter_search():
    """Example: Hyperparameter search for USFlow."""
    print("\nExample 7: USFlow Hyperparameter Search")
    print("="*60)
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/usflow_hypersearch"),
        device="cuda",
        verbose=False,
    )
    
    # USFlow-specific parameter grid
    param_grid = {
        'flow_type': ['usflow'],
        'coupling_blocks': [4, 8, 12],
        'hidden_dim': [64, 128, 256],
        'lr': [1e-4, 1e-3, 1e-2],
        'batch_size': [32, 64],
        'lu_transform': [0, 1, 2],
        'householder': [0, 1],
        'affine_conjugation': [True, False],
        'masktype': ['checkerboard', 'channel'],
        'epochs': [100],
        'patience': [10],
    }
    
    tune_datasets = ["6_cardio", "2_annthyroid"]
    
    benchmark = ADBenchBenchmark(config)
    results_df = benchmark.hyperparameter_search(
        datasets=tune_datasets,
        param_grid=param_grid,
        n_trials=15,
    )
    
    print("\nTop 5 USFlow configurations by ROC-AUC:")
    top_configs = results_df.nlargest(5, 'roc_auc')
    print(top_configs[[
        'dataset', 'roc_auc', 'pr_auc',
        'config_coupling_blocks', 'config_lu_transform', 
        'config_householder', 'config_masktype'
    ]])


def example_usflow_with_convnet():
    """Example: Run USFlow with ConvNet conditioner."""
    print("\nExample 8: USFlow with ConvNet Conditioner")
    print("="*60)
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/usflow_convnet"),
        device="cuda",
        verbose=True,
    )
    
    # Configure USFlow with ConvNet - specify c_hidden directly
    flow_config = FlowConfig(
        flow_type="usflow",
        coupling_blocks=8,
        c_hidden=[256, 128, 64],  # Direct specification of hidden layers
        lr=1e-3,
        batch_size=64,
        epochs=100,
        patience=10,
        lu_transform=1,
        householder=0,
        affine_conjugation=True,
        prior_scale=1.0,
        masktype="checkerboard",
    )
    
    # Select datasets to test
    test_datasets = [
        "6_cardio",
        "2_annthyroid",
        "38_thyroid",
    ]
    
    benchmark = ADBenchBenchmark(config)
    results = benchmark.run_on_datasets(
        datasets=test_datasets,
        flow_config=flow_config,
    )
    
    benchmark.save_results("usflow_convnet_benchmark")
    print("\nUSFlow with ConvNet Summary:")
    print(benchmark.summary())


def example_nonusflow_with_custom_architecture():
    """Example: Run NonUSFlow with custom conditioner architecture."""
    print("\nExample 9: NonUSFlow with Custom Architecture")
    print("="*60)
    
    config = BenchmarkConfig(
        data_dir=Path("./data/adbench"),
        output_dir=Path("./results/adbench/nonusflow_custom"),
        device="cuda",
        verbose=True,
    )
    
    # Deep architecture with gradually decreasing hidden dimensions
    flow_config = FlowConfig(
        flow_type="nonusflow",
        coupling_blocks=10,
        c_hidden=[512, 256, 128, 64],  # 4-layer deep conditioner
        lr=1e-3,
        batch_size=64,
        epochs=100,
        patience=10,
        clamp=5.0,
        affine_conjugation=True,
    )
    
    test_datasets = ["6_cardio", "4_breastw", "38_thyroid"]
    
    benchmark = ADBenchBenchmark(config)
    results = benchmark.run_on_datasets(
        datasets=test_datasets,
        flow_config=flow_config,
    )
    
    benchmark.save_results("nonusflow_custom_arch")
    print("\nNonUSFlow Custom Architecture Summary:")
    print(benchmark.summary())

if __name__ == '__main__':
    # Run examples
    example_single_dataset()
    example_multiple_datasets()
    example_hyperparameter_search()
    
    # USFlow examples
    try:
        example_usflow_benchmark()
        example_compare_flows()
        example_usflow_hyperparameter_search()
        example_usflow_with_convnet()
        example_nonusflow_with_custom_architecture()
    except ImportError as e:
        print(f"\nSkipping USFlow examples: {e}")
    
    # Uncomment to run full benchmark (takes hours!)
    # example_full_benchmark()
