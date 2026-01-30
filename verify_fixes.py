import inspect
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

# Mock mcp module
from unittest.mock import MagicMock

# Create a mock that acts as a pass-through decorator for @mcp.tool()
def pass_through_decorator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

mcp_mock = MagicMock()
mcp_mock.tool.side_effect = pass_through_decorator

# Configure sys.modules
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()
# Ensure FastMCP returns our configured mock
sys.modules["mcp.server.fastmcp"].FastMCP.return_value = mcp_mock

sys.modules["mcp.server.session"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["joblib"] = MagicMock()
sys.modules["numpy"] = MagicMock()

def test_path_utils():
    print("Testing path_utils...")
    from src.core.path_utils import get_output_dir, to_absolute_path, resolve_data_path
    
    # Test get_output_dir construction
    out = get_output_dir("/tmp/test_work_dir", "reports")
    assert str(out) == "/tmp/test_work_dir/reports"
    print("✅ get_output_dir works")

def test_server_signatures():
    print("Testing server tool signatures...")
    from src.server import train_automatic, run_eda
    
    # Check train_automatic
    sig = inspect.signature(train_automatic)
    params = sig.parameters
    assert 'output_dir' in params, "train_automatic missing output_dir"
    assert 'timeout_minutes' in params, "train_automatic missing timeout_minutes"
    print("✅ train_automatic signature correct")
    
    # Check run_eda
    sig = inspect.signature(run_eda)
    params = sig.parameters
    assert 'output_dir' in params, "run_eda missing output_dir"
    print("✅ run_eda signature correct")

    # Check ping tool
    from src.server import ping
    assert inspect.iscoroutinefunction(ping), "ping must be async"
    print("✅ ping tool exists and is async")

def test_pipeline_signatures():
    print("Testing pipeline signatures...")
    from src.training.pipeline import run_automatic_training, run_benchmark_training
    from src.eda.pipeline import run_eda_pipeline
    
    # Check run_automatic_training
    sig = inspect.signature(run_automatic_training)
    assert 'output_dir' in sig.parameters, "run_automatic_training missing output_dir"
    print("✅ run_automatic_training signature correct")
    
    # Check run_benchmark_training
    sig = inspect.signature(run_benchmark_training)
    assert 'output_dir' in sig.parameters, "run_benchmark_training missing output_dir"
    print("✅ run_benchmark_training signature correct")

    # Check run_eda_pipeline
    sig = inspect.signature(run_eda_pipeline)
    assert 'output_dir' in sig.parameters, "run_eda_pipeline missing output_dir"
    print("✅ run_eda_pipeline signature correct")

if __name__ == "__main__":
    try:
        test_path_utils()
        test_server_signatures()
        test_pipeline_signatures()
        print("\n🎉 All verification tests passed!")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        sys.exit(1)
    except AssertionError as e:
        print(f"❌ Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)
