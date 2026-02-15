---
name: pdf-to-images
description: Converts research paper PDFs into high-quality, high-resolution images (one per page). Uses Python with pdf2image and Pillow for maximum quality output.
---

# PDF to Images Skill

## Purpose
This skill converts research paper PDFs into individual high-resolution images, one per page. This is useful for:
- Creating visual previews of papers
- Generating thumbnails for presentations
- Sharing paper pages on social media or websites
- Creating image-based documentation
- Archiving papers in image format

## When to Use
- After generating a LaTeX paper PDF (e.g., using the latex-paper-generator skill)
- When the user requests to convert a PDF to images
- When preparing papers for visual presentation or sharing
- When creating a gallery of research outputs

## How It Works
The skill uses Python's `pdf2image` library (which wraps `poppler-utils`) to convert PDF pages to high-quality PNG images. Each page is rendered at high DPI (300+) to ensure crisp, publication-quality output.

## Prerequisites
The skill will automatically check for and install required dependencies:
- `poppler-utils` (system package for PDF rendering)
- `pdf2image` (Python library)
- `Pillow` (Python imaging library)

## Usage Instructions

### Step 1: Identify the PDF
Locate the PDF file you want to convert. Common locations:
- `/root/llm-research-kit/docs/papers/*.pdf`
- Any custom path provided by the user

### Step 2: Run the Conversion Script
Execute the conversion script with the PDF path:

```bash
python .agent/skills/pdf-to-images/scripts/convert_pdf.py <pdf_path> [output_dir] [--dpi DPI]
```

**Arguments:**
- `pdf_path`: Path to the PDF file (required)
- `output_dir`: Directory to save images (optional, defaults to `<pdf_dir>/images/<pdf_name>`)
- `--dpi`: Resolution in DPI (optional, default: 300, recommended: 300-600)

**Example:**
```bash
python .agent/skills/pdf-to-images/scripts/convert_pdf.py docs/papers/paper.pdf
```

This will create images in `docs/papers/images/paper/` named `page_001.png`, `page_002.png`, etc.

### Step 3: Verify Output
Check the output directory to ensure all pages were converted successfully. The script will report:
- Number of pages converted
- Output directory location
- File names and sizes

### Step 4: Report Results
Inform the user of:
- Number of pages converted
- Output location
- Image resolution (DPI)
- Total file size
- Any errors or warnings

## Quality Settings

### DPI Recommendations
- **300 DPI**: Standard print quality, good for most uses (~1-2 MB per page)
- **450 DPI**: High quality, excellent for detailed figures (~2-4 MB per page)
- **600 DPI**: Maximum quality, for archival or large prints (~4-8 MB per page)

### Output Format
- **PNG**: Lossless compression, preserves all details (default)
- Alternative formats can be added if needed (JPEG, TIFF)

## Error Handling
If the script encounters errors:
1. **Missing dependencies**: Install `poppler-utils` and Python packages
2. **PDF not found**: Verify the path is correct
3. **Permission errors**: Check file/directory permissions
4. **Memory issues**: Reduce DPI or process pages in batches

## Implementation Notes
- The script processes pages sequentially to avoid memory issues
- Progress is reported for long PDFs
- Images are named with zero-padded page numbers for proper sorting
- Existing images are overwritten by default (add `--no-overwrite` flag if needed)

## Example Workflow
```bash
# 1. Generate a LaTeX paper (using latex-paper-generator skill)
# 2. Convert the resulting PDF to images
python .agent/skills/pdf-to-images/scripts/convert_pdf.py docs/papers/paper.pdf --dpi 450

# 3. Verify output
ls -lh docs/papers/images/paper/
```

## Customization
The script can be extended to:
- Add watermarks to images
- Crop margins automatically
- Generate thumbnails alongside full-size images
- Convert to different formats (JPEG, WebP)
- Batch process multiple PDFs
