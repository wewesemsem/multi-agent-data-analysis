"""Export Drafting Agent artifacts to PDF / DOCX.

Structured draft artifacts remain the canonical source. This module is a
presentation layer only — it does not recalculate analysis or regenerate charts.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from app.state import SharedWorkspace


# ---------------------------------------------------------------------------
# Normalized export document (shared by PDF + DOCX)
# ---------------------------------------------------------------------------


@dataclass
class ExportTable:
    headers: list[str]
    rows: list[list[str]]
    caption: str = ""


@dataclass
class ExportImage:
    png_bytes: bytes
    caption: str = ""
    chart_id: str | None = None


@dataclass
class ExportBlock:
    kind: str  # heading | paragraph | bullets | numbered | table | image | note
    text: str = ""
    level: int = 1
    items: list[str] = field(default_factory=list)
    table: ExportTable | None = None
    image: ExportImage | None = None


@dataclass
class ExportDocument:
    title: str
    draft_type: str
    blocks: list[ExportBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # non-fatal export warnings


def safe_export_filename(title: str, ext: str) -> str:
    """Derive a filesystem-safe download filename from an artifact title."""
    base = (title or "draft").strip()
    base = re.sub(r"[^\w\s\-]+", "", base, flags=re.UNICODE)
    base = re.sub(r"[\s\-]+", "_", base).strip("_")
    if not base:
        base = "draft"
    base = base[:80]
    ext = ext.lstrip(".").lower()
    if ext not in {"pdf", "docx"}:
        ext = "pdf"
    return f"{base}.{ext}"


def build_export_document(
    draft: dict[str, Any],
    workspace: SharedWorkspace | None = None,
) -> ExportDocument:
    """Normalize a draft artifact (+ workspace viz lookup) into export blocks."""
    title = str(draft.get("title") or "Draft").strip() or "Draft"
    draft_type = str(draft.get("type") or "summary")
    doc = ExportDocument(title=title, draft_type=draft_type)
    seen_tables: set[str] = set()
    embedded_chart_ids: set[str] = set()

    # Title handled by exporters; start with type note
    doc.blocks.append(
        ExportBlock(kind="paragraph", text=f"Artifact type: {draft_type}")
    )

    sections = list(draft.get("sections") or [])
    if sections:
        for sec in sections:
            heading = str(sec.get("heading") or "Section").strip()
            kind_label = str(sec.get("kind") or "").strip()
            body = str(sec.get("body") or "").strip()
            doc.blocks.append(ExportBlock(kind="heading", text=heading, level=1))
            if kind_label:
                doc.blocks.append(
                    ExportBlock(kind="paragraph", text=f"({kind_label})")
                )
            doc.blocks.extend(_blocks_from_markdown(body, seen_tables))
            for table in sec.get("tables") or []:
                et = _coerce_table(table)
                if et:
                    key = _table_key(et)
                    if key not in seen_tables:
                        seen_tables.add(key)
                        doc.blocks.append(ExportBlock(kind="table", table=et))
            for chart_ref in sec.get("charts") or []:
                img, warn = _resolve_chart_image(chart_ref, workspace, draft)
                if warn:
                    doc.warnings.append(warn)
                if img and (img.chart_id or img.caption) not in embedded_chart_ids:
                    embedded_chart_ids.add(img.chart_id or img.caption)
                    doc.blocks.append(ExportBlock(kind="image", image=img))
    else:
        # Fall back to full markdown content
        content = str(draft.get("content") or "").strip()
        if content:
            doc.blocks.extend(_blocks_from_markdown(content, seen_tables))

    # Top-level structured tables
    for table in draft.get("tables") or []:
        et = _coerce_table(table)
        if et:
            key = _table_key(et)
            if key not in seen_tables:
                seen_tables.add(key)
                doc.blocks.append(ExportBlock(kind="table", table=et))

    # Charts referenced on the draft
    chart_refs: list[Any] = list(draft.get("charts") or [])
    for ref in draft.get("source_artifacts") or []:
        if isinstance(ref, dict) and ref.get("kind") == "visualization":
            chart_refs.append(ref.get("id") or ref.get("index"))
    for ref in chart_refs:
        img, warn = _resolve_chart_image(ref, workspace, draft)
        if warn:
            doc.warnings.append(warn)
        if img:
            key = img.chart_id or img.caption
            if key in embedded_chart_ids:
                continue
            embedded_chart_ids.add(key)
            doc.blocks.append(ExportBlock(kind="image", image=img))

    assumptions = _as_str_list(draft.get("assumptions"))
    warnings = _as_str_list(draft.get("warnings"))
    limitations = _as_str_list(draft.get("limitations"))
    if assumptions:
        doc.blocks.append(ExportBlock(kind="heading", text="Assumptions", level=1))
        doc.blocks.append(ExportBlock(kind="bullets", items=assumptions))
    if warnings:
        doc.blocks.append(ExportBlock(kind="heading", text="Warnings", level=1))
        doc.blocks.append(ExportBlock(kind="bullets", items=warnings))
    if limitations:
        doc.blocks.append(ExportBlock(kind="heading", text="Limitations", level=1))
        doc.blocks.append(ExportBlock(kind="bullets", items=limitations))

    sources = _format_sources(draft.get("source_artifacts") or [], workspace)
    if sources:
        doc.blocks.append(ExportBlock(kind="heading", text="Sources", level=1))
        doc.blocks.append(ExportBlock(kind="bullets", items=sources))

    return doc


def export_draft_pdf(
    draft: dict[str, Any],
    workspace: SharedWorkspace | None = None,
) -> bytes:
    """Render a draft artifact to PDF bytes (selectable text)."""
    doc = build_export_document(draft, workspace)
    return _render_pdf(doc)


def export_draft_docx(
    draft: dict[str, Any],
    workspace: SharedWorkspace | None = None,
) -> bytes:
    """Render a draft artifact to DOCX bytes (editable)."""
    doc = build_export_document(draft, workspace)
    return _render_docx(doc)


# ---------------------------------------------------------------------------
# Markdown / structure parsing
# ---------------------------------------------------------------------------


_MD_TABLE_ROW = re.compile(r"^\|(.+)\|$")
_MD_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_MD_UL = re.compile(r"^[-*+]\s+(.+)$")
_MD_OL = re.compile(r"^(\d+)[.)]\s+(.+)$")


def _blocks_from_markdown(text: str, seen_tables: set[str]) -> list[ExportBlock]:
    if not text:
        return []
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[ExportBlock] = []
    i = 0
    para_buf: list[str] = []

    def flush_para() -> None:
        nonlocal para_buf
        if not para_buf:
            return
        joined = " ".join(p.strip() for p in para_buf if p.strip())
        if joined:
            # Keep inline markdown markers for PDF/DOCX converters.
            blocks.append(ExportBlock(kind="paragraph", text=joined))
        para_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_para()
            i += 1
            continue

        # Markdown table
        if _MD_TABLE_ROW.match(stripped) and i + 1 < len(lines) and _MD_TABLE_SEP.match(lines[i + 1].strip()):
            flush_para()
            header = _split_md_row(stripped)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and _MD_TABLE_ROW.match(lines[i].strip()):
                rows.append(_split_md_row(lines[i].strip()))
                i += 1
            # Cap very large tables
            if len(rows) > 200:
                rows = rows[:200]
            et = ExportTable(headers=header, rows=rows)
            key = _table_key(et)
            if key not in seen_tables:
                seen_tables.add(key)
                blocks.append(ExportBlock(kind="table", table=et))
            continue

        hm = _MD_HEADING.match(stripped)
        if hm:
            flush_para()
            level = min(len(hm.group(1)), 3)
            blocks.append(
                ExportBlock(
                    kind="heading",
                    text=hm.group(2).strip(),
                    level=level,
                )
            )
            i += 1
            continue

        if stripped in {"---", "***", "___"}:
            flush_para()
            i += 1
            continue

        um = _MD_UL.match(stripped)
        if um:
            flush_para()
            items = [um.group(1).strip()]
            i += 1
            while i < len(lines):
                um2 = _MD_UL.match(lines[i].strip())
                if not um2:
                    break
                items.append(um2.group(1).strip())
                i += 1
            blocks.append(ExportBlock(kind="bullets", items=items))
            continue

        om = _MD_OL.match(stripped)
        if om:
            flush_para()
            items = [om.group(2).strip()]
            i += 1
            while i < len(lines):
                om2 = _MD_OL.match(lines[i].strip())
                if not om2:
                    break
                items.append(om2.group(2).strip())
                i += 1
            blocks.append(ExportBlock(kind="numbered", items=items))
            continue

        para_buf.append(stripped)
        i += 1

    flush_para()
    return blocks


def _split_md_row(line: str) -> list[str]:
    inner = line.strip().strip("|")
    return [_strip_md_to_plainish(c.strip()) for c in inner.split("|")]


def _strip_md_to_plainish(text: str) -> str:
    """Remove common markdown markers while keeping readable text for exporters."""
    s = text or ""
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return s.strip()


def _md_inline_to_reportlab(text: str) -> str:
    """Convert a subset of markdown inline markers to ReportLab rich-text tags."""
    s = escape(text or "")
    # Restore intentional placeholders then apply formatting on raw, safer path:
    s = text or ""
    # Escape first, then wrap known patterns from original via staged replace
    # Work on escaped version of content with markers preserved carefully:
    parts: list[str] = []
    # Simpler: escape, then apply bold/italic on escaped text patterns that used markers
    # Re-parse from original:
    out = ""
    i = 0
    raw = text or ""
    while i < len(raw):
        if raw.startswith("**", i):
            end = raw.find("**", i + 2)
            if end != -1:
                out += f"<b>{escape(raw[i+2:end])}</b>"
                i = end + 2
                continue
        if raw.startswith("__", i):
            end = raw.find("__", i + 2)
            if end != -1:
                out += f"<b>{escape(raw[i+2:end])}</b>"
                i = end + 2
                continue
        if raw.startswith("`", i):
            end = raw.find("`", i + 1)
            if end != -1:
                out += f"<font face='Courier'>{escape(raw[i+1:end])}</font>"
                i = end + 1
                continue
        if raw[i] == "*" and i + 1 < len(raw) and raw[i + 1] != "*":
            end = raw.find("*", i + 1)
            if end != -1 and (end == 0 or raw[end - 1] != "*"):
                out += f"<i>{escape(raw[i+1:end])}</i>"
                i = end + 1
                continue
        if raw[i] == "_" and i + 1 < len(raw) and raw[i + 1] != "_":
            end = raw.find("_", i + 1)
            if end != -1:
                out += f"<i>{escape(raw[i+1:end])}</i>"
                i = end + 1
                continue
        out += escape(raw[i])
        i += 1
    return out.replace("\n", "<br/>")


# ---------------------------------------------------------------------------
# Tables / sources / charts
# ---------------------------------------------------------------------------


def _coerce_table(raw: Any) -> ExportTable | None:
    if not isinstance(raw, dict):
        return None
    headers = raw.get("headers") or raw.get("columns")
    rows = raw.get("rows") or raw.get("data")
    if headers and rows is not None:
        h = [str(x) for x in headers]
        out_rows: list[list[str]] = []
        for row in list(rows)[:200]:
            if isinstance(row, dict):
                out_rows.append([_cell(row.get(col)) for col in h])
            elif isinstance(row, (list, tuple)):
                out_rows.append([_cell(v) for v in row])
            else:
                out_rows.append([_cell(row)])
        return ExportTable(headers=h, rows=out_rows, caption=str(raw.get("caption") or ""))
    # records: list[dict]
    records = raw.get("records")
    if isinstance(records, list) and records and isinstance(records[0], dict):
        h = list(records[0].keys())
        out_rows = [[_cell(r.get(c)) for c in h] for r in records[:200] if isinstance(r, dict)]
        return ExportTable(headers=[str(x) for x in h], rows=out_rows, caption=str(raw.get("caption") or ""))
    return None


def _cell(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _table_key(table: ExportTable) -> str:
    return "|".join(table.headers) + "||" + ";;".join(",".join(r) for r in table.rows[:5])


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return []


def _format_sources(
    refs: list[Any],
    workspace: SharedWorkspace | None,
) -> list[str]:
    lines: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        kind = ref.get("kind")
        idx = ref.get("index")
        label = None
        if workspace is not None and isinstance(idx, int):
            if kind == "analysis" and 0 <= idx < len(workspace.analysis_results or []):
                item = workspace.analysis_results[idx]
                label = (
                    item.get("question")
                    or (item.get("query_plan") or {}).get("mode")
                    or f"Analysis[{idx}]"
                )
                lines.append(f"Analysis result: {label}")
                continue
            if kind == "forecast" and 0 <= idx < len(workspace.forecasts or []):
                item = workspace.forecasts[idx]
                target = item.get("target_column") or "series"
                method = item.get("selected_method") or "forecast"
                lines.append(f"Forecast result: {target} ({method})")
                continue
            if kind == "anomaly" and 0 <= idx < len(workspace.anomalies or []):
                item = workspace.anomalies[idx]
                lines.append(
                    f"Anomaly result: {item.get('method')} on {item.get('column')} "
                    f"({item.get('n_anomalies')} flagged)"
                )
                continue
            if kind == "visualization" and 0 <= idx < len(workspace.visualizations or []):
                item = workspace.visualizations[idx]
                lines.append(
                    f"Visualization: {item.get('title') or item.get('id')} "
                    f"({item.get('chart_type')})"
                )
                continue
        if kind:
            extra = ref.get("id") or (f"index={idx}" if idx is not None else "")
            lines.append(f"{kind.replace('_', ' ').title()} result: {extra}".strip())
    # de-dupe preserve order
    out: list[str] = []
    for line in lines:
        if line not in out:
            out.append(line)
    return out


def _resolve_chart_image(
    ref: Any,
    workspace: SharedWorkspace | None,
    draft: dict[str, Any],
) -> tuple[ExportImage | None, str | None]:
    viz = _lookup_visualization(ref, workspace)
    if viz is None:
        return None, None
    caption = str(viz.get("title") or viz.get("id") or "Chart")
    chart_id = viz.get("id")
    try:
        png = _visualization_to_png(viz)
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            f"Chart '{caption}' could not be embedded ({type(exc).__name__}: {exc}).",
        )
    if not png:
        return None, f"Chart '{caption}' could not be embedded; text export continued."
    return ExportImage(png_bytes=png, caption=caption, chart_id=str(chart_id) if chart_id else None), None


def _lookup_visualization(ref: Any, workspace: SharedWorkspace | None) -> dict[str, Any] | None:
    if workspace is None:
        return None
    viz_list = workspace.visualizations or []
    if not viz_list:
        return None
    if isinstance(ref, dict):
        if ref.get("id"):
            for v in viz_list:
                if v.get("id") == ref.get("id"):
                    return v
        idx = ref.get("index")
        if isinstance(idx, int) and 0 <= idx < len(viz_list):
            return viz_list[idx]
        return None
    if isinstance(ref, int):
        if 0 <= ref < len(viz_list):
            return viz_list[ref]
        return None
    if isinstance(ref, str):
        for v in viz_list:
            if v.get("id") == ref:
                return v
    return None


def _visualization_to_png(viz: dict[str, Any]) -> bytes | None:
    """Best-effort chart image from an existing visualization artifact."""
    # 1) Plotly + kaleido (when browser engine works)
    fig_json = viz.get("plotly_json")
    if not fig_json and viz.get("json_path"):
        try:
            import json

            fig_json = json.loads(Path(viz["json_path"]).read_text())
        except Exception:  # noqa: BLE001
            fig_json = None
    if fig_json:
        try:
            import plotly.io as pio

            payload = fig_json if isinstance(fig_json, str) else __import__("json").dumps(fig_json)
            fig = pio.from_json(payload)
            return fig.to_image(format="png", width=900, height=500, scale=2)
        except Exception:  # noqa: BLE001
            pass

    # 2) Pillow fallback from data_preview (no external browser)
    return _preview_chart_png(viz)


def _preview_chart_png(viz: dict[str, Any]) -> bytes | None:
    preview = viz.get("data_preview") or []
    if not preview or not isinstance(preview, list):
        return None
    x_key = viz.get("x")
    y_key = viz.get("y")
    rows = [r for r in preview if isinstance(r, dict)]
    if not rows:
        return None
    if not x_key or x_key not in rows[0]:
        x_key = next(iter(rows[0].keys()), None)
    if not y_key or y_key not in rows[0]:
        # pick first numeric-looking column that isn't x
        y_key = None
        for k, v in rows[0].items():
            if k == x_key:
                continue
            try:
                float(v)
                y_key = k
                break
            except (TypeError, ValueError):
                continue
    if not x_key or not y_key:
        return None

    xs = [str(r.get(x_key, ""))[:16] for r in rows[:40]]
    ys: list[float] = []
    for r in rows[:40]:
        try:
            ys.append(float(r.get(y_key)))
        except (TypeError, ValueError):
            ys.append(0.0)
    if not ys:
        return None

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # noqa: BLE001
        return None

    width, height = 900, 500
    margin = 60
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    title = str(viz.get("title") or "Chart")
    draw.text((margin, 16), title[:80], fill="black", font=font)

    plot_left, plot_top = margin, margin + 10
    plot_right, plot_bottom = width - margin, height - margin
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=1)

    min_y, max_y = min(ys), max(ys)
    if abs(max_y - min_y) < 1e-12:
        max_y = min_y + 1.0
    n = max(len(ys) - 1, 1)
    points = []
    for i, val in enumerate(ys):
        px = plot_left + (plot_right - plot_left) * (i / n)
        py = plot_bottom - (plot_bottom - plot_top) * ((val - min_y) / (max_y - min_y))
        points.append((px, py))
    if len(points) >= 2:
        draw.line(points, fill="#1f4e79", width=2)
    for px, py in points:
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill="#1f4e79")

    # axis labels (sparse)
    if xs:
        draw.text((plot_left, plot_bottom + 8), xs[0], fill="gray", font=font)
        draw.text((plot_right - 40, plot_bottom + 8), xs[-1], fill="gray", font=font)
    draw.text((8, plot_top), f"{max_y:.4g}", fill="gray", font=font)
    draw.text((8, plot_bottom - 12), f"{min_y:.4g}", fill="gray", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF (reportlab)
# ---------------------------------------------------------------------------


def _render_pdf(doc: ExportDocument) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=doc.title,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ExportTitle",
            parent=styles["Title"],
            fontSize=18,
            spaceAfter=12,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ExportH1",
            parent=styles["Heading1"],
            fontSize=14,
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ExportH2",
            parent=styles["Heading2"],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ExportBody",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ExportCaption",
            parent=styles["BodyText"],
            fontSize=9,
            textColor=colors.grey,
            spaceAfter=10,
        )
    )

    story: list[Any] = [Paragraph(escape(doc.title), styles["ExportTitle"])]
    tmp_files: list[Path] = []

    try:
        for block in doc.blocks:
            if block.kind == "heading":
                style = styles["ExportH1"] if block.level <= 1 else styles["ExportH2"]
                story.append(Paragraph(escape(block.text), style))
            elif block.kind == "paragraph":
                # Prefer inline markdown conversion from original-ish text
                story.append(Paragraph(_md_inline_to_reportlab(block.text), styles["ExportBody"]))
            elif block.kind in {"bullets", "numbered"}:
                items = []
                for item in block.items:
                    items.append(
                        ListItem(Paragraph(_md_inline_to_reportlab(item), styles["ExportBody"]))
                    )
                story.append(
                    ListFlowable(
                        items,
                        bulletType="1" if block.kind == "numbered" else "bullet",
                        start="1",
                    )
                )
                story.append(Spacer(1, 6))
            elif block.kind == "table" and block.table:
                data = [block.table.headers] + block.table.rows
                # Keep cells as Paragraphs for wrapping
                wrapped = [
                    [Paragraph(escape(str(c)), styles["ExportBody"]) for c in row] for row in data
                ]
                col_count = max(len(r) for r in wrapped) if wrapped else 1
                usable = LETTER[0] - 1.5 * inch
                col_w = usable / max(col_count, 1)
                tbl = Table(wrapped, colWidths=[col_w] * col_count, repeatRows=1)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(tbl)
                if block.table.caption:
                    story.append(Paragraph(escape(block.table.caption), styles["ExportCaption"]))
                story.append(Spacer(1, 8))
            elif block.kind == "image" and block.image:
                fd, tmp_name = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                tmp = Path(tmp_name)
                tmp.write_bytes(block.image.png_bytes)
                tmp_files.append(tmp)
                img = RLImage(str(tmp))
                img.drawWidth = 6.5 * inch
                img.drawHeight = 3.6 * inch
                story.append(img)
                if block.image.caption:
                    story.append(Paragraph(escape(block.image.caption), styles["ExportCaption"]))
            elif block.kind == "note" and block.text:
                story.append(Paragraph(escape(block.text), styles["ExportCaption"]))

        for warn in doc.warnings:
            story.append(Paragraph(escape(f"Note: {warn}"), styles["ExportCaption"]))

        pdf.build(story)
        return buffer.getvalue()
    finally:
        for path in tmp_files:
            try:
                path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# DOCX (python-docx)
# ---------------------------------------------------------------------------


def _render_docx(doc: ExportDocument) -> bytes:
    from docx import Document
    from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
    from docx.shared import Inches, Pt

    document = Document()
    title = document.add_heading(doc.title, level=0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    for block in doc.blocks:
        if block.kind == "heading":
            level = 1 if block.level <= 1 else min(block.level, 3)
            document.add_heading(block.text, level=level)
        elif block.kind == "paragraph":
            p = document.add_paragraph()
            _add_runs_with_basic_md(p, block.text)
        elif block.kind == "bullets":
            for item in block.items:
                p = document.add_paragraph(style="List Bullet")
                _add_runs_with_basic_md(p, item)
        elif block.kind == "numbered":
            for item in block.items:
                p = document.add_paragraph(style="List Number")
                _add_runs_with_basic_md(p, item)
        elif block.kind == "table" and block.table:
            cols = len(block.table.headers) or 1
            rows = 1 + len(block.table.rows)
            table = document.add_table(rows=rows, cols=cols)
            table.style = "Table Grid"
            for j, h in enumerate(block.table.headers):
                table.rows[0].cells[j].text = str(h)
            for i, row in enumerate(block.table.rows, start=1):
                for j in range(cols):
                    val = row[j] if j < len(row) else ""
                    table.rows[i].cells[j].text = str(val)
            if block.table.caption:
                cap = document.add_paragraph(block.table.caption)
                for run in cap.runs:
                    run.font.size = Pt(9)
                    run.italic = True
        elif block.kind == "image" and block.image:
            try:
                stream = io.BytesIO(block.image.png_bytes)
                document.add_picture(stream, width=Inches(6.0))
                if block.image.caption:
                    cap = document.add_paragraph(block.image.caption)
                    for run in cap.runs:
                        run.font.size = Pt(9)
                        run.italic = True
            except Exception:  # noqa: BLE001
                document.add_paragraph(f"[Chart could not be embedded: {block.image.caption}]")

    for warn in doc.warnings:
        p = document.add_paragraph(f"Note: {warn}")
        for run in p.runs:
            run.font.size = Pt(9)
            run.italic = True

    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def _add_runs_with_basic_md(paragraph: Any, text: str) -> None:
    """Apply a minimal bold/italic subset into a python-docx paragraph."""
    # Clear default empty run if present
    if paragraph.runs:
        paragraph.runs[0].text = ""
    raw = text or ""
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|[^*`]+)")
    for token in pattern.findall(raw) or [raw]:
        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*") and token.endswith("*") and len(token) >= 2:
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        elif token.startswith("`") and token.endswith("`") and len(token) >= 2:
            run = paragraph.add_run(token[1:-1])
        else:
            paragraph.add_run(_strip_md_to_plainish(token) if ("_" in token or "*" in token) else token)
