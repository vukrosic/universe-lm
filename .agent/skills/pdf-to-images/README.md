# PDF to Images Skill

Converts research paper PDFs into high-quality, high-resolution images (one per page).

## Quick Start

```bash
# Convert a PDF with default settings (300 DPI)
python .agent/skills/pdf-to-images/scripts/convert_pdf.py docs/papers/paper.pdf

# Convert with higher resolution (450 DPI)
python .agent/skills/pdf-to-images/scripts/convert_pdf.py docs/papers/paper.pdf --dpi 450

# Convert to a specific directory
python .agent/skills/pdf-to-images/scripts/convert_pdf.py docs/papers/paper.pdf ./my_images
```

## Features

- **High Quality**: Generates PNG images at 300+ DPI for crisp, publication-quality output
- **Automatic Setup**: Installs all required dependencies automatically
- **Progress Tracking**: Shows conversion progress and file sizes
- **Flexible Output**: Customizable output directory and resolution
- **Batch Processing**: Efficiently processes multi-page PDFs

## Output

Images are saved as `page_001.png`, `page_002.png`, etc. in the output directory.

Default output location: `<pdf_directory>/images/<pdf_name>/`

## Requirements

The script automatically installs:
- `poppler-utils` (system package)
- `pdf2image` (Python library)
- `Pillow` (Python library)

## See Also

- **SKILL.md**: Complete usage instructions and guidelines
- **latex-paper-generator**: Skill for generating LaTeX PDFs
