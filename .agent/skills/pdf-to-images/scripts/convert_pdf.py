#!/usr/bin/env python3
"""
PDF to Images Converter
Converts PDF pages to high-quality, high-resolution PNG images.

Usage:
    python convert_pdf.py <pdf_path> [output_dir] [--dpi DPI] [--no-overwrite]

Examples:
    python convert_pdf.py paper.pdf
    python convert_pdf.py paper.pdf ./images --dpi 450
    python convert_pdf.py paper.pdf --dpi 600 --no-overwrite
"""

import argparse
import os
import sys
from pathlib import Path


def check_dependencies():
    """Check and install required dependencies."""
    print("Checking dependencies...")
    
    # Check for poppler-utils
    poppler_check = os.system("which pdftoppm > /dev/null 2>&1")
    if poppler_check != 0:
        print("Installing poppler-utils...")
        result = os.system("apt-get update -qq && apt-get install -y -qq poppler-utils")
        if result != 0:
            print("ERROR: Failed to install poppler-utils")
            sys.exit(1)
    
    # Check for Python packages
    try:
        import pdf2image
        from PIL import Image
        print("✓ All dependencies are installed")
    except ImportError as e:
        print(f"Installing Python packages: {e.name}")
        result = os.system("pip install -q pdf2image Pillow")
        if result != 0:
            print("ERROR: Failed to install Python packages")
            sys.exit(1)
        print("✓ Python packages installed")


def convert_pdf_to_images(pdf_path, output_dir=None, dpi=300, overwrite=True):
    """
    Convert PDF to high-quality images.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images (default: auto-generated)
        dpi: Resolution in DPI (default: 300)
        overwrite: Whether to overwrite existing images (default: True)
    
    Returns:
        List of paths to generated images
    """
    from pdf2image import convert_from_path
    from PIL import Image
    
    # Validate PDF path
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Determine output directory
    if output_dir is None:
        pdf_name = pdf_path.stem
        output_dir = pdf_path.parent / "images" / pdf_name
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PDF to Images Converter")
    print(f"{'='*60}")
    print(f"Input PDF:     {pdf_path}")
    print(f"Output Dir:    {output_dir}")
    print(f"Resolution:    {dpi} DPI")
    print(f"{'='*60}\n")
    
    # Convert PDF to images
    print("Converting PDF pages to images...")
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt='png',
            thread_count=4,  # Use multiple threads for faster conversion
        )
    except Exception as e:
        print(f"ERROR: Failed to convert PDF: {e}")
        sys.exit(1)
    
    # Save images
    output_paths = []
    total_size = 0
    
    for i, image in enumerate(images, start=1):
        # Generate filename with zero-padded page number
        filename = f"page_{i:03d}.png"
        output_path = output_dir / filename
        
        # Check if file exists and overwrite is disabled
        if output_path.exists() and not overwrite:
            print(f"  Skipping page {i}/{len(images)}: {filename} (already exists)")
            output_paths.append(output_path)
            continue
        
        # Save image with maximum quality
        image.save(
            str(output_path),
            'PNG',
            optimize=True,  # Optimize PNG compression
            compress_level=6,  # Balance between speed and compression
        )
        
        # Get file size
        file_size = output_path.stat().st_size
        total_size += file_size
        
        # Report progress
        size_mb = file_size / (1024 * 1024)
        print(f"  ✓ Page {i}/{len(images)}: {filename} ({size_mb:.2f} MB)")
        
        output_paths.append(output_path)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"{'='*60}")
    print(f"Pages converted:  {len(images)}")
    print(f"Total size:       {total_size / (1024 * 1024):.2f} MB")
    print(f"Average per page: {total_size / len(images) / (1024 * 1024):.2f} MB")
    print(f"Output location:  {output_dir}")
    print(f"{'='*60}\n")
    
    return output_paths


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert PDF pages to high-quality PNG images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf
  %(prog)s paper.pdf ./images --dpi 450
  %(prog)s paper.pdf --dpi 600 --no-overwrite
        """
    )
    
    parser.add_argument(
        'pdf_path',
        help='Path to the PDF file'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?',
        default=None,
        help='Output directory (default: <pdf_dir>/images/<pdf_name>)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution in DPI (default: 300, recommended: 300-600)'
    )
    
    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Skip existing images instead of overwriting'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Convert PDF
    try:
        output_paths = convert_pdf_to_images(
            pdf_path=args.pdf_path,
            output_dir=args.output_dir,
            dpi=args.dpi,
            overwrite=not args.no_overwrite
        )
        
        print(f"✓ Successfully converted {len(output_paths)} pages")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
