"""
PDF export service for conversation health reports (ReportLab).
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime
from datetime import timezone
from io import BytesIO
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..config import PDF_INCLUDE_METADATA, PDF_INCLUDE_PROFILE, PDF_PAGE_SIZE
from .conversation_store import ConversationStore
from .models import Conversation, MessageRole, PatientProfile
from .profile_manager import ProfileManager

logger = logging.getLogger(__name__)


class ExportService:
    """Generate clinical-style PDF reports from stored conversations."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        profile_manager: ProfileManager,
    ):
        self.conversation_store = conversation_store
        self.profile_manager = profile_manager
        self._styles = self._build_styles()

    def _page_size(self):
        return A4 if PDF_PAGE_SIZE == "a4" else letter

    def _build_styles(self):
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Heading1"],
                alignment=TA_CENTER,
                fontSize=16,
                spaceAfter=6,
                textColor=colors.HexColor("#0f172a"),
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionHead",
                parent=styles["Heading2"],
                fontSize=12,
                spaceBefore=12,
                spaceAfter=6,
                textColor=colors.HexColor("#0e7490"),
            )
        )
        styles.add(
            ParagraphStyle(
                name="BodyTextCustom",
                parent=styles["BodyText"],
                fontSize=9,
                leading=12,
                alignment=TA_LEFT,
            )
        )
        styles.add(
            ParagraphStyle(
                name="EmergencyText",
                parent=styles["BodyText"],
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#b91c1c"),
                backColor=colors.HexColor("#fef2f2"),
                borderPadding=4,
            )
        )
        styles.add(
            ParagraphStyle(
                name="FooterNote",
                parent=styles["Normal"],
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_CENTER,
            )
        )
        return styles

    def generate_pdf(
        self,
        conversation_id: str,
        patient_id: str,
        include_profile: Optional[bool] = None,
    ) -> BytesIO:
        """Build a PDF report; raises FileNotFoundError if conversation missing."""
        conversation = self.conversation_store.load_conversation(
            conversation_id, patient_id
        )
        if conversation is None:
            raise FileNotFoundError(
                f"Conversation {conversation_id} not found for patient {patient_id}"
            )

        include = PDF_INCLUDE_PROFILE if include_profile is None else include_profile
        profile = (
            self.profile_manager.load_profile(patient_id) if include else None
        )

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self._page_size(),
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
            title=f"Conversation Report {conversation_id}",
        )
        story: List = []
        story.extend(self._build_header(conversation, profile))
        if include:
            story.extend(self._build_profile_section(profile))
        story.extend(self._build_messages_section(conversation))
        story.extend(self._build_footer())
        doc.build(story)
        buffer.seek(0)
        return buffer

    def _build_header(
        self,
        conversation: Conversation,
        profile: Optional[PatientProfile],
    ) -> List:
        elements: List = [
            Paragraph("CONVERSATION HEALTH REPORT", self._styles["ReportTitle"]),
            Paragraph("AuraHealth AI Assistant", self._styles["FooterNote"]),
            Spacer(1, 8),
        ]
        if not PDF_INCLUDE_METADATA:
            return elements

        meta = conversation.metadata
        timestamps = [m.timestamp for m in conversation.messages] or [
            meta.created_at,
            meta.updated_at,
        ]
        date_start = min(timestamps).strftime("%B %d, %Y %I:%M %p")
        date_end = max(timestamps).strftime("%B %d, %Y %I:%M %p")
        generated = datetime.now(timezone.utc).strftime("%B %d, %Y %I:%M %p")

        rows = [
            ["Patient ID", self._sanitize_content(meta.patient_id)],
            ["Conversation ID", self._sanitize_content(meta.conversation_id)],
            ["Title", self._sanitize_content(meta.title or "Untitled")],
            ["Date Range", f"{date_start} – {date_end}"],
            ["Report Generated", generated],
        ]
        table = Table(rows, colWidths=[1.6 * inch, 5.0 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0f2fe")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(table)
        return elements

    def _build_profile_section(self, profile: Optional[PatientProfile]) -> List:
        elements: List = [Paragraph("PATIENT PROFILE SUMMARY", self._styles["SectionHead"])]
        if profile is None:
            elements.append(
                Paragraph("No patient profile on file.", self._styles["BodyTextCustom"])
            )
            return elements

        diagnoses = ", ".join(d.condition_name for d in profile.diagnoses) or "None"
        medications = (
            ", ".join(f"{m.name} {m.dosage}" for m in profile.medications) or "None"
        )
        allergies = (
            ", ".join(f"{a.allergen} ({a.severity})" for a in profile.allergies) or "None"
        )
        name = profile.name or "Unknown"
        age = f" (Age {profile.age})" if profile.age is not None else ""

        for line in [
            f"<b>Name:</b> {self._sanitize_content(name)}{age}",
            f"<b>Diagnoses:</b> {self._sanitize_content(diagnoses)}",
            f"<b>Medications:</b> {self._sanitize_content(medications)}",
            f"<b>Allergies:</b> {self._sanitize_content(allergies)}",
        ]:
            elements.append(Paragraph(line, self._styles["BodyTextCustom"]))
        return elements

    def _build_messages_section(self, conversation: Conversation) -> List:
        elements: List = [
            Paragraph("CONVERSATION TRANSCRIPT", self._styles["SectionHead"])
        ]
        if not conversation.messages:
            elements.append(
                Paragraph("No messages in this conversation.", self._styles["BodyTextCustom"])
            )
            return elements

        for msg in conversation.messages:
            ts = msg.timestamp.strftime("%I:%M %p")
            if msg.role == MessageRole.USER:
                sender = "Patient"
            elif msg.role == MessageRole.ASSISTANT:
                sender = "AuraHealth Assistant"
            else:
                sender = "System"

            emergency = False
            if isinstance(msg.metadata, dict):
                emerg = msg.metadata.get("emergency") or {}
                if isinstance(emerg, dict) and emerg.get("is_emergency"):
                    emergency = True

            header = f"[{ts}] {sender}:"
            body = self._sanitize_content(msg.content)
            if emergency:
                elements.append(
                    Paragraph(
                        f"⚠ CLINICAL ALERT — {header}<br/>{body}",
                        self._styles["EmergencyText"],
                    )
                )
            else:
                elements.append(
                    Paragraph(f"<b>{header}</b><br/>{body}", self._styles["BodyTextCustom"])
                )
            elements.append(Spacer(1, 6))
        return elements

    def _build_footer(self) -> List:
        return [
            Spacer(1, 16),
            Paragraph("DISCLAIMER", self._styles["SectionHead"]),
            Paragraph(
                "This conversation report is for informational purposes only and does not "
                "constitute medical advice. Always consult with your healthcare provider "
                "before making any changes to your treatment.",
                self._styles["FooterNote"],
            ),
        ]

    def _sanitize_content(self, text: str) -> str:
        """Escape HTML/XML entities and strip unsupported control characters for PDF."""
        if text is None:
            return ""
        cleaned = str(text)
        # Remove control chars except newline/tab
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
        # Escape for ReportLab Paragraph (XML-ish)
        cleaned = html.escape(cleaned)
        cleaned = cleaned.replace("\n", "<br/>")
        return cleaned
