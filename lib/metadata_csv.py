"""
🎨 Violet Model Merge - Metadata Manager Library

Clean, focused library for CSV-based SafeTensors metadata management.
Perfect for batch editing model metadata with external tools!

Author: Violet Tools
Version: 1.2.1
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime


def install_dependencies():
    """Install required dependencies if missing."""
    import subprocess
    import sys
    
    try:
        import pandas as pd
    except ImportError:
        print("📦 Installing Pandas...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        print("✅ Pandas installed!")
    
    try:
        import safetensors
        import safetensors.torch
    except ImportError:
        print("📦 Installing SafeTensors...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors"])
        print("✅ SafeTensors installed!")


def scan_safetensors_files(directory):
    """Scan directory for .safetensors files only (excludes .sha256, .txt, etc.)"""
    if not os.path.exists(directory):
        return []
    
    files = []
    for file in os.listdir(directory):
        # 🛡️ STRICT FILTER: Only .safetensors files, exclude checksums and other files
        if file.lower().endswith('.safetensors') and not file.lower().endswith('.sha256'):
            files.append(file)
    return sorted(files)


def verify_paths(models_path, vae_path, output_path):
    """Verify that paths exist and create output directory."""
    status = {"models": False, "vae": False}
    
    # Create output directory
    Path(output_path).mkdir(exist_ok=True)
    
    if os.path.exists(models_path):
        print(f"✅ Models directory verified: {models_path}")
        status["models"] = True
    else:
        print(f"⚠️ Models path doesn't exist: {models_path}")
    
    if os.path.exists(vae_path):
        print(f"✅ VAE directory verified: {vae_path}")
        status["vae"] = True
    else:
        print(f"⚠️ VAE path doesn't exist: {vae_path}")
    
    print(f"📁 Output directory: {output_path}")
    return status


def read_safetensors_metadata(file_path):
    """Read metadata from a SafeTensors file."""
    import safetensors
    
    try:
        with safetensors.safe_open(file_path, framework="pt") as f:
            metadata = f.metadata()
            return metadata if metadata else {}
    except Exception as e:
        print(f"❌ Error reading metadata from {file_path}: {e}")
        return {}


def get_file_info(file_path):
    """Get basic file information including size and modification date."""
    try:
        stat = os.stat(file_path)
        return {
            "size_gb": round(stat.st_size / (1024**3), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size_bytes": stat.st_size
        }
    except Exception as e:
        print(f"⚠️ Could not get file info for {file_path}: {e}")
        return {}


def load_model_metadata(filename, models_path, vae_path, is_vae=False):
    """Load comprehensive metadata for a model or VAE file."""
    base_path = vae_path if is_vae else models_path
    file_path = os.path.join(base_path, filename)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {filename}"}
    
    # Get all information
    metadata = read_safetensors_metadata(file_path)
    file_info = get_file_info(file_path)
    
    return {
        "filename": filename,
        "file_type": "VAE" if is_vae else "Model",
        "file_path": file_path,
        "file_info": file_info,
        "metadata": metadata,
        "has_metadata": len(metadata) > 0
    }


def export_models_to_csv(models_list, models_path, vae_path, output_path, filename=None):
    """Export model metadata to an editable CSV file."""
    import pandas as pd
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"editable_metadata_{timestamp}.csv"
    
    csv_path = Path(output_path) / filename
    
    # 🛡️ SAFETY FILTER: Only process .safetensors files
    models_list = [f for f in models_list if f.lower().endswith('.safetensors')]
    
    if not models_list:
        print("❌ No .safetensors models to export")
        return None
    
    print(f"📊 Exporting metadata for {len(models_list)} .safetensors models to CSV...")
    
    # Load metadata for all selected models
    csv_data = []
    
    for filename_model in models_list:
        print(f"📖 Loading: {filename_model}")
        model_info = load_model_metadata(filename_model, models_path, vae_path, is_vae=False)
        
        if 'error' in model_info:
            print(f"   ⚠️ Skipping due to error: {model_info['error']}")
            continue
        
        # Prepare row data
        metadata = model_info.get('metadata', {})
        file_info = model_info.get('file_info', {})
        
        row = {
            'filename': filename_model,
            'file_size_gb': file_info.get('size_gb', ''),
            'modified_date': file_info.get('modified', ''),
        }
        
        # Common metadata fields that users often want to edit
        common_fields = [
            'description', 'author', 'version', 'memo', 'tags', 
            'base_model', 'merge_method', 'alpha', 'beta',
            'training_info', 'license', 'usage_notes'
        ]
        
        # Add common fields (with current values or empty)
        for field in common_fields:
            row[field] = metadata.get(field, '')
        
        # Add any existing metadata fields not in common list
        for key, value in metadata.items():
            if key not in common_fields and key not in row:
                row[key] = value
        
        csv_data.append(row)
    
    if not csv_data:
        print("❌ No valid models to export")
        return None
    
    # Create DataFrame and save to CSV
    try:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\n✅ CSV exported successfully!")
        print(f"📁 File: {filename}")
        print(f"📊 Rows: {len(df)} models")
        print(f"📋 Columns: {len(df.columns)} fields")
        
        return str(csv_path)
        
    except Exception as e:
        print(f"❌ Failed to create CSV: {e}")
        return None


def import_csv_to_models(csv_filename, output_path, models_path, dry_run=True, create_backups=False):
    """Compatibility wrapper that delegates to the Windows-safe implementation.

    This ensures a single, reliable code path is used across platforms and
    avoids file-handle issues common on Windows. The behavior is identical
    to the previous function, but implemented via the copy-edit-replace flow.
    """
    return import_csv_to_models_windows_safe(
        csv_filename=csv_filename,
        output_path=output_path,
        models_path=models_path,
        dry_run=dry_run,
        create_backups=create_backups,
    )


def import_csv_to_models_windows_safe(csv_filename, output_path, models_path, dry_run=True, create_backups=False):
    """Windows-safe version that handles file locks with copy-edit-replace method."""
    import pandas as pd
    import safetensors
    import safetensors.torch
    import tempfile
    import shutil
    import gc

    csv_path = Path(output_path) / csv_filename

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_filename}")
        return False

    print(f"📥 Windows-safe import from: {csv_filename}")

    try:
        # Read the CSV
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"📊 Found {len(df)} models in CSV")

        # Process each row to identify changes
        changes_summary = []
        for index, row in df.iterrows():
            filename = row.get('filename', '')
            if not filename:
                continue

            file_path = os.path.join(models_path, filename)
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {filename}, skipping")
                continue

            # Load current metadata
            current_metadata = read_safetensors_metadata(file_path)

            # Prepare new metadata from CSV row
            new_metadata = {}
            changes_for_file = []

            skip_columns = ['filename', 'file_size_gb', 'modified_date']
            for column, value in row.items():
                if column in skip_columns or pd.isna(value) or value == '':
                    continue

                clean_value = str(value).strip()
                if not clean_value:
                    continue

                new_metadata[column] = clean_value

                old_value = current_metadata.get(column, '[NOT SET]')
                if old_value != clean_value:
                    changes_for_file.append(f"  • {column}: '{old_value}' → '{clean_value}'")

            if changes_for_file:
                changes_summary.append({
                    'filename': filename,
                    'file_path': file_path,
                    'changes': changes_for_file,
                    'new_metadata': {**current_metadata, **new_metadata}
                })

        # Show summary
        print(f"\n📋 IMPORT SUMMARY")
        print(f"{'='*50}")
        print(f"Files with changes: {len(changes_summary)}")

        if not changes_summary:
            print("✅ No changes detected!")
            return True

        for change_info in changes_summary:
            print(f"\n📄 {change_info['filename']}:")
            for change in change_info['changes']:
                print(change)

        if dry_run:
            print(f"\n🧪 DRY RUN MODE - No changes applied")
            return True

        # Windows-safe application using temp directory
        print(f"\n🚀 Applying changes using Windows-safe method...")
        print(f"🔧 Using copy-edit-replace to avoid file locks")

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="violet_metadata_"))

        try:
            success_count = 0

            for change_info in changes_summary:
                filename = change_info['filename']
                original_path = Path(change_info['file_path'])
                new_metadata = change_info['new_metadata']

                # Create temp copy
                temp_path = temp_dir / filename
                shutil.copy2(original_path, temp_path)

                try:
                    # Force garbage collection to release any handles
                    gc.collect()

                    # Edit the temp copy
                    with safetensors.safe_open(str(temp_path), framework="pt") as f:
                        # Clone tensors to fully detach from any underlying mmap
                        tensors = {key: f.get_tensor(key).clone() for key in f.keys()}

                    # Clear the file handle
                    del f
                    gc.collect()

                    # Save with new metadata to temp file
                    safetensors.torch.save_file(tensors, str(temp_path), metadata=new_metadata)

                    # Clear tensors from memory
                    del tensors
                    gc.collect()

                    # Create backup if requested
                    if create_backups:
                        backup_path = Path(output_path) / f"{original_path.stem}_backup{original_path.suffix}"
                        shutil.copy2(original_path, backup_path)

                    # Replace original with edited version
                    shutil.copy2(temp_path, original_path)

                    print(f"✅ Updated: {filename}")
                    success_count += 1

                except Exception as e:
                    print(f"❌ Failed to update {filename}: {e}")

            print(f"\n🎉 Windows-safe import complete!")
            print(f"   ✅ Successfully updated: {success_count}")
            print(f"   ❌ Failed: {len(changes_summary) - success_count}")

            return success_count == len(changes_summary)

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"❌ Failed to import CSV: {e}")
        return False


def backup_original_file(file_path, backup_dir):
    """Create a backup of the original file before modifying."""
    file_path = Path(file_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}_backup{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    try:
        shutil.copy2(file_path, backup_path)
        print(f"💾 Backup created: {backup_name}")
        return str(backup_path)
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return None


def display_metadata_summary(model_info):
    """Display a beautiful summary of model metadata."""
    if "error" in model_info:
        print(f"❌ {model_info['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"📄 {model_info['filename']} ({model_info['file_type']})")
    print(f"{'='*60}")
    
    # File information
    file_info = model_info.get('file_info', {})
    if file_info:
        print(f"📊 Size: {file_info.get('size_gb', '?')} GB")
        print(f"📅 Modified: {file_info.get('modified', 'Unknown')}")
    
    # Metadata
    metadata = model_info.get('metadata', {})
    print(f"\n🏷️ Metadata Fields: {len(metadata)}")
    
    if not metadata:
        print("   📭 No metadata found")
        return
    
    # Show common fields first
    common_fields = [
        'memo', 'description', 'author', 'version', 'merge_method',
        'base_model', 'training_info'
    ]
    
    found_common = []
    for field in common_fields:
        if field in metadata:
            found_common.append(field)
            value = metadata[field]
            if isinstance(value, str) and len(value) > 80:
                value = value[:77] + "..."
            print(f"   • {field}: {value}")
    
    # Show other fields
    other_fields = [k for k in metadata.keys() if k not in found_common]
    if other_fields:
        print(f"\n📌 Other Fields ({len(other_fields)}):")
        for field in sorted(other_fields)[:10]:  # Show first 10
            value = metadata[field]
            if isinstance(value, str) and len(value) > 60:
                value = value[:57] + "..."
            print(f"   • {field}: {value}")
        
        if len(other_fields) > 10:
            print(f"   ... and {len(other_fields) - 10} more fields")


def quick_setup(models_path, vae_path, output_path):
    """Quick setup function to initialize the metadata manager."""
    print("🎨 Violet Model Merge - Metadata Manager")
    print("="*50)
    
    # Verify paths
    path_status = verify_paths(models_path, vae_path, output_path)
    
    # Scan for files
    available_models = scan_safetensors_files(models_path) if path_status["models"] else []
    available_vaes = scan_safetensors_files(vae_path) if path_status["vae"] else []
    
    print(f"\n📊 Found {len(available_models)} model files and {len(available_vaes)} VAE files")
    print("🎯 Ready for CSV-based metadata management!")
    
    return {
        "path_status": path_status,
        "available_models": available_models,
        "available_vaes": available_vaes,
        "models_path": models_path,
        "vae_path": vae_path,
        "output_path": output_path
    }