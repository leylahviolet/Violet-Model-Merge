#!/usr/bin/env python3
"""
🧪 Violet Model Merge - Local Test Runner

A convenient script for running tests locally with various options.
This script mimics the CI environment for local development.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description, check=True, capture_output=False):
    """Run a command with proper error handling."""
    print(f"\n🔧 {description}")
    print(f"💻 Running: {' '.join(cmd)}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result.stdout, result.stderr
        else:
            subprocess.run(cmd, check=check)
            return None, None
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error output: {e.stderr}")
        if not check:
            print("⚠️ Continuing despite failure...")
        else:
            sys.exit(1)


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pytest', 'pytest-cov', 'torch', 'safetensors', 'numpy', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install " + " ".join(missing))
        return False
    
    return True


def run_tests(args):
    """Run the test suite with specified options."""
    if not check_dependencies():
        if not args.ignore_deps:
            sys.exit(1)
        print("⚠️ Continuing with missing dependencies...")
    
    # Base pytest command
    cmd = ['python', '-m', 'pytest', 'tests/']
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            '--cov=lib',
            '--cov-report=html:tests/htmlcov',
            '--cov-report=term-missing',
            '--cov-report=xml'
        ])
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    elif args.quiet:
        cmd.append('-q')
    
    # Test selection
    if args.unit_only:
        cmd.extend(['-m', 'unit'])
    elif args.integration_only:
        cmd.extend(['-m', 'integration'])
    elif args.fast:
        cmd.extend(['-m', 'not slow and not gpu'])
    
    # Parallel execution
    if args.parallel and args.parallel > 1:
        cmd.extend(['-n', str(args.parallel)])
    
    # Output format
    if args.html_report:
        cmd.extend(['--html=test-report.html', '--self-contained-html'])
    
    if args.junit_xml:
        cmd.extend(['--junit-xml=test-results.xml'])
    
    # Additional pytest options
    if args.fail_fast:
        cmd.append('-x')
    
    if args.tb_format:
        cmd.extend(['--tb', args.tb_format])
    
    # Run the tests
    run_command(cmd, "Running test suite")
    
    # Print coverage summary if coverage was run
    if args.coverage:
        print("\n📊 Coverage report generated in tests/htmlcov/")


def run_linting(args):
    """Run code quality checks."""
    print("\n🔍 Running code quality checks...")
    
    # Black formatting check
    if not args.no_format:
        run_command([
            'python', '-m', 'black', '--check', '--diff', 'lib/', 'tests/'
        ], "Checking code formatting with Black", check=False)
    
    # Import sorting check
    if not args.no_imports:
        run_command([
            'python', '-m', 'isort', '--check-only', '--diff', 'lib/', 'tests/'
        ], "Checking import sorting with isort", check=False)
    
    # Flake8 linting
    if not args.no_lint:
        run_command([
            'python', '-m', 'flake8', 'lib/', 'tests/',
            '--max-line-length=88',
            '--extend-ignore=E203,W503'
        ], "Linting with flake8", check=False)
    
    # Type checking with mypy
    if not args.no_types:
        run_command([
            'python', '-m', 'mypy', 'lib/',
            '--ignore-missing-imports'
        ], "Type checking with mypy", check=False)


def install_dev_deps():
    """Install development dependencies."""
    print("📦 Installing development dependencies...")
    
    dev_packages = [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'pytest-xdist>=2.5.0',
        'pytest-html>=3.1.0',
        'pytest-mock>=3.7.0',
        'black>=22.0.0',
        'isort>=5.10.0',
        'flake8>=4.0.0',
        'mypy>=0.950',
        'bandit>=1.7.0',
        'safety>=2.0.0'
    ]
    
    cmd = ['python', '-m', 'pip', 'install'] + dev_packages
    run_command(cmd, "Installing development dependencies")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🧪 Violet Model Merge - Local Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                     # Run all tests with coverage
  python run_tests.py --fast              # Run only fast tests
  python run_tests.py --unit-only         # Run only unit tests
  python run_tests.py --lint              # Run code quality checks
  python run_tests.py --install-deps      # Install development dependencies
        """
    )
    
    # Main actions
    parser.add_argument('--install-deps', action='store_true',
                       help='Install development dependencies')
    parser.add_argument('--lint', action='store_true',
                       help='Run code quality checks')
    
    # Test selection
    parser.add_argument('--unit-only', action='store_true',
                       help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--fast', action='store_true',
                       help='Run only fast tests (skip slow and GPU tests)')
    
    # Test options
    parser.add_argument('--coverage', action='store_true', default=True,
                       help='Generate coverage report (default: True)')
    parser.add_argument('--no-coverage', dest='coverage', action='store_false',
                       help='Skip coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output')
    parser.add_argument('--parallel', '-n', type=int, metavar='N',
                       help='Run tests in parallel with N workers')
    parser.add_argument('--fail-fast', '-x', action='store_true',
                       help='Stop on first failure')
    parser.add_argument('--tb-format', choices=['short', 'long', 'auto', 'line', 'native'],
                       default='short', help='Traceback format')
    
    # Output options
    parser.add_argument('--html-report', action='store_true',
                       help='Generate HTML test report')
    parser.add_argument('--junit-xml', action='store_true',
                       help='Generate JUnit XML report')
    
    # Lint options
    parser.add_argument('--no-format', action='store_true',
                       help='Skip formatting check')
    parser.add_argument('--no-imports', action='store_true',
                       help='Skip import sorting check')
    parser.add_argument('--no-lint', action='store_true',
                       help='Skip flake8 linting')
    parser.add_argument('--no-types', action='store_true',
                       help='Skip type checking')
    
    # Utility options
    parser.add_argument('--ignore-deps', action='store_true',
                       help='Continue even if dependencies are missing')
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🧪 Violet Model Merge - Local Test Runner")
    print(f"📁 Working directory: {project_root}")
    
    # Execute requested actions
    if args.install_deps:
        install_dev_deps()
        return
    
    if args.lint:
        run_linting(args)
        return
    
    # Default action: run tests
    run_tests(args)
    
    print("\n✅ Test run completed!")


if __name__ == '__main__':
    main()