#!/bin/bash
#
# Run Marine Algorithm Benchmarks
# Compare Python vs Rust performance
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║  🏁  MARINE ALGORITHM PERFORMANCE SHOWDOWN  🏁           ║"
echo "║                                                           ║"
echo "║  Python 🐍  vs  Rust 🦀                                  ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Clean previous results
echo "🧹 Cleaning previous results..."
rm -f python_benchmark_results.json rust_benchmark_results.json

# Build Rust benchmark
echo ""
echo "🔨 Building Rust benchmark (opt-level=3, lto=true)..."
cargo build --release
echo "   ✅ Rust benchmark built!"

# Run Python benchmark
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Running Python benchmark..."
echo "═══════════════════════════════════════════════════════════"
python3 marine_benchmark.py

# Run Rust benchmark
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Running Rust benchmark..."
echo "═══════════════════════════════════════════════════════════"
./target/release/marine_benchmark

# Compare results
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📊 COMPARISON REPORT"
echo "═══════════════════════════════════════════════════════════"

# Parse and compare JSON results
python3 - <<'EOF'
import json

print("\nLoading results...")
with open('python_benchmark_results.json') as f:
    py_results = json.load(f)

with open('rust_benchmark_results.json') as f:
    rust_results = json.load(f)

print("\n" + "="*63)
print("DETAILED COMPARISON")
print("="*63)
print(f"{'Test':<20} {'Python (Ms/s)':<15} {'Rust (Ms/s)':<15} {'Speedup':<10}")
print("-"*63)

speedups = []
for py_test, rust_test in zip(py_results['tests'], rust_results['tests']):
    py_throughput = py_test['throughput_msamples']
    rust_throughput = rust_test['throughput_msamples']
    speedup = rust_throughput / py_throughput
    speedups.append(speedup)

    print(f"{py_test['name']:<20} {py_throughput:<15.2f} {rust_throughput:<15.2f} {speedup:<10.2f}x")

print("-"*63)
avg_speedup = sum(speedups) / len(speedups)
print(f"{'AVERAGE':<20} {'':<15} {'':<15} {avg_speedup:<10.2f}x")
print("="*63)

print("\n╔═══════════════════════════════════════════════════════════╗")
print("║  VERDICT                                                  ║")
print("╚═══════════════════════════════════════════════════════════╝")

if avg_speedup > 2.0:
    print(f"  🦀 Rust is {avg_speedup:.1f}x FASTER than Python! 🚀")
    print(f"     Every clock cycle optimized!")
elif avg_speedup > 1.5:
    print(f"  🦀 Rust wins with {avg_speedup:.1f}x speedup! ⚡")
    print(f"     Demoscene-worthy performance!")
else:
    print(f"  Both implementations are fast!")
    print(f"  Rust: {avg_speedup:.1f}x advantage")

print("\n📊 Detailed results saved to:")
print("   • python_benchmark_results.json")
print("   • rust_benchmark_results.json")

EOF

echo ""
echo "✅ Benchmark complete!"
