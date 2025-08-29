"""
Test runner script for trajectory prediction system.

This script provides utilities for running different types of tests:
- Unit tests with coverage reporting
- Integration tests 
- Performance benchmarks
- Custom test suites
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time
from typing import List, Dict, Any, Optional

def run_command(cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error output: {result.stderr}")
    
    return result


def run_unit_tests(
    coverage: bool = True,
    verbose: bool = True,
    parallel: bool = False,
    pattern: Optional[str] = None
) -> bool:
    """Run unit tests with optional coverage reporting."""
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if pattern:
        cmd.extend(["-k", pattern])
    else:
        cmd.append("tests/unit/")
    
    # Coverage options
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html:htmlcov", 
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=85"
        ])
    
    # Output options
    if verbose:
        cmd.append("--verbose")
    else:
        cmd.append("--quiet")
    
    # Parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])  # Requires pytest-xdist
    
    # Additional options
    cmd.extend([
        "--tb=short",
        "--durations=10",
        "--strict-markers"
    ])
    
    result = run_command(cmd, capture_output=False)
    return result.returncode == 0


def run_integration_tests(
    verbose: bool = True,
    pattern: Optional[str] = None
) -> bool:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if pattern:
        cmd.extend(["-k", pattern])
    else:
        cmd.append("tests/integration/")
    
    # Options
    if verbose:
        cmd.append("--verbose")
    
    cmd.extend([
        "--tb=short",
        "--durations=10",
        "--strict-markers"
    ])
    
    result = run_command(cmd, capture_output=False)
    return result.returncode == 0


def run_performance_tests(
    verbose: bool = True,
    pattern: Optional[str] = None,
    benchmark_only: bool = False
) -> bool:
    """Run performance tests and benchmarks."""
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if pattern:
        cmd.extend(["-k", pattern])
    elif benchmark_only:
        cmd.extend(["-m", "performance"])
    else:
        cmd.append("tests/performance/")
    
    # Options for performance tests
    cmd.extend([
        "--verbose" if verbose else "--quiet",
        "--tb=short",
        "--durations=0",  # Show all durations
        "--benchmark-disable-gc",  # Disable GC during benchmarks
        "--benchmark-warmup=on",   # Enable warmup
        "--strict-markers"
    ])
    
    result = run_command(cmd, capture_output=False)
    return result.returncode == 0


def run_all_tests(
    coverage: bool = True,
    verbose: bool = True,
    fast: bool = False
) -> Dict[str, bool]:
    """Run all test suites in sequence."""
    results = {}
    
    print("=" * 80)
    print("Running Unit Tests")
    print("=" * 80)
    
    results['unit'] = run_unit_tests(
        coverage=coverage,
        verbose=verbose,
        parallel=fast
    )
    
    if results['unit']:
        print("\n" + "=" * 80)
        print("Running Integration Tests")
        print("=" * 80)
        
        results['integration'] = run_integration_tests(verbose=verbose)
        
        if not fast:  # Skip performance tests in fast mode
            print("\n" + "=" * 80)
            print("Running Performance Tests")
            print("=" * 80)
            
            results['performance'] = run_performance_tests(verbose=verbose)
        else:
            results['performance'] = True  # Skip but mark as passed
    else:
        print("Unit tests failed, skipping integration and performance tests")
        results['integration'] = False
        results['performance'] = False
    
    return results


def run_specific_test(
    test_path: str,
    verbose: bool = True,
    coverage: bool = False
) -> bool:
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("--verbose")
    
    cmd.extend(["--tb=short", "--strict-markers"])
    
    result = run_command(cmd, capture_output=False)
    return result.returncode == 0


def run_test_with_markers(
    markers: List[str],
    verbose: bool = True
) -> bool:
    """Run tests with specific markers."""
    cmd = ["python", "-m", "pytest"]
    
    # Build marker expression
    if len(markers) == 1:
        marker_expr = markers[0]
    else:
        marker_expr = " or ".join(markers)
    
    cmd.extend(["-m", marker_expr])
    
    if verbose:
        cmd.append("--verbose")
    
    cmd.extend(["--tb=short", "--strict-markers"])
    
    result = run_command(cmd, capture_output=False)
    return result.returncode == 0


def check_dependencies() -> bool:
    """Check that required test dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-asyncio",
        "hypothesis"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required test dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def generate_test_report(results: Dict[str, bool]) -> None:
    """Generate a summary test report."""
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    total_suites = len(results)
    passed_suites = sum(1 for passed in results.values() if passed)
    
    for suite_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{suite_name.title()} Tests: {status}")
    
    print(f"\nOverall: {passed_suites}/{total_suites} test suites passed")
    
    if passed_suites == total_suites:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Test runner for trajectory prediction system")
    
    parser.add_argument(
        "suite",
        nargs="?",
        choices=["unit", "integration", "performance", "all"],
        default="all",
        help="Test suite to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true", 
        help="Reduce test output verbosity"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run tests in fast mode (parallel, skip performance)"
    )
    
    parser.add_argument(
        "--pattern", "-k",
        help="Pattern to filter tests"
    )
    
    parser.add_argument(
        "--markers", "-m",
        nargs="+",
        help="Run tests with specific markers"
    )
    
    parser.add_argument(
        "--test",
        help="Run specific test file or function"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps or not check_dependencies():
        return 1 if args.check_deps else 1
    
    start_time = time.time()
    
    try:
        # Handle specific test execution
        if args.test:
            success = run_specific_test(
                args.test,
                verbose=not args.quiet,
                coverage=not args.no_coverage
            )
            return 0 if success else 1
        
        # Handle marker-based execution  
        if args.markers:
            success = run_test_with_markers(
                args.markers,
                verbose=not args.quiet
            )
            return 0 if success else 1
        
        # Handle suite execution
        if args.suite == "unit":
            success = run_unit_tests(
                coverage=not args.no_coverage,
                verbose=not args.quiet,
                parallel=args.fast,
                pattern=args.pattern
            )
            return 0 if success else 1
            
        elif args.suite == "integration":
            success = run_integration_tests(
                verbose=not args.quiet,
                pattern=args.pattern
            )
            return 0 if success else 1
            
        elif args.suite == "performance":
            success = run_performance_tests(
                verbose=not args.quiet,
                pattern=args.pattern,
                benchmark_only=True
            )
            return 0 if success else 1
            
        elif args.suite == "all":
            results = run_all_tests(
                coverage=not args.no_coverage,
                verbose=not args.quiet,
                fast=args.fast
            )
            
            # Generate summary report
            success = generate_test_report(results)
            return 0 if success else 1
    
    finally:
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    sys.exit(main())