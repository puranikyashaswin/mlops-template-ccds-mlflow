import subprocess
import sys

def test_train_script():
    # Run the training script
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        capture_output=True,
        text=True,
        cwd="."
    )
    # Assert exit code 0
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    # Check for expected output
    assert "Training complete" in result.stdout
    assert "MAE:" in result.stdout
    assert "R2 Score:" in result.stdout
    assert "Run ID:" in result.stdout
