from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import logging
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field


SEED = 20261101
SCHEMA_VERSION = "1.0.0"

STATUS_COLORS = {
    "accepted": (46, 204, 113),
    "review": (241, 196, 15),
    "candidate": (127, 140, 141),
    "rejected": (120, 120, 120),
    "contradicted": (231, 76, 60),
}

TYPE_COLORS = {
    "PERSON": (52, 152, 219),
    "ORG": (155, 89, 182),
    "PRODUCT": (26, 188, 156),
    "GPE": (241, 196, 15),
    "UNKNOWN": (160, 160, 160),
}

logger = logging.getLogger("cdri-demo")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def log_event(event: str, **kwargs: Any) -> None:
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    payload.update(kwargs)
    logger.info(json.dumps(payload, sort_keys=True, default=str))


class BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_version: str = SCHEMA_VERSION


class PipelineSettings(BaseSchema):
    seed: int = SEED
    extractor_version: str = "cdri-rule-v1"
    min_confidence: float = Field(0.35, ge=0.0, le=1.0)
    simulation_steps: int = Field(8, ge=1, le=50)
    video_fps: int = Field(12, ge=1, le=30)
    video_seconds: float = Field(2.5, ge=0.5, le=10.0)
    canvas_width: int = Field(640, ge=320, le=1280)
    canvas_height: int = Field(360, ge=180, le=720)
    auto_generate_video: bool = True
    fixed_ingestion_timestamp: str = "2026-11-01T00:00:00+00:00"

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"schema_version"})
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class SourceDocument(BaseSchema):
    doc_id: str
    title: str
    text: str
    source: str
    published: str
    sha256: str


class Mention(BaseSchema):
    mention_id: str
    doc_id: str
    entity_id: str
    entity_label: str
    entity_type: str
    span_start: int
    span_end: int
    excerpt: str
    extractor_version: str
    confidence: float
    ingested_at: str


class Relation(BaseSchema):
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    status: Literal["candidate", "review", "accepted", "rejected", "contradicted"] = "candidate"
    evidence_mention_ids: list[str] = Field(default_factory=list)
    evidence_docs: list[str] = Field(default_factory=list)
    evidence_spans: list[dict[str, Any]] = Field(default_factory=list)
    extractor_version: str
    ingested_at: str
    contradiction_notes: list[str] = Field(default_factory=list)


_DEMO_DOCS: list[dict[str, str]] = [
    {
        "doc_id": "doc-001",
        "title": "Helios Energy Announces Grid AI Pilot",
        "source": "Utility Wire",
        "published": "2026-08-14",
        "text": (
            "Dr. Mara Chen, chief scientist at Helios Energy, announced a pilot with National Grid "
            "to deploy predictive maintenance AI. Chen said the project will reduce outages. "
            "The pilot begins in Austin."
        ),
    },
    {
        "doc_id": "doc-002",
        "title": "Regulators Scrutinize AI Maintenance Claims",
        "source": "Grid Policy Daily",
        "published": "2026-08-19",
        "text": (
            "A federal reviewer warned that Helios Energy lacks independent validation for its "
            "predictive maintenance AI. Dr. Mara Chen responded that National Grid will audit the pilot. "
            "The reviewer cited earlier outage data from Austin."
        ),
    },
    {
        "doc_id": "doc-003",
        "title": "Helios Pauses Austin Pilot After Safety Review",
        "source": "Regional Energy Report",
        "published": "2026-08-27",
        "text": (
            "Helios Energy has paused the Austin pilot while an external safety review continues. "
            "National Grid confirmed the pause. Dr. Mara Chen was unavailable for comment."
        ),
    },
    {
        "doc_id": "doc-004",
        "title": "Industry Group Backs Cross-Document Evidence Standards",
        "source": "Smart Grid News",
        "published": "2026-09-02",
        "text": (
            "The Smart Grid Institute endorsed a cross-document evidence standard proposed by Dr. Mara Chen. "
            "Helios Energy and National Grid joined the working group. The institute said transparency reduces risk."
        ),
    },
]

ENTITY_ALIASES: dict[str, tuple[str, list[str]]] = {
    "Dr. Mara Chen": ("PERSON", ["Dr. Mara Chen", "Mara Chen", "Chen"]),
    "Helios Energy": ("ORG", ["Helios Energy", "Helios"]),
    "National Grid": ("ORG", ["National Grid"]),
    "Smart Grid Institute": ("ORG", ["Smart Grid Institute", "institute"]),
    "Austin": ("GPE", ["Austin"]),
    "predictive maintenance AI": (
        "PRODUCT",
        ["predictive maintenance AI", "maintenance AI", "pilot", "pause"],
    ),
}

RELATION_TRIGGERS: dict[str, str] = {
    "announced": "announced_initiative",
    "begins": "announced_initiative",
    "launch": "announced_initiative",
    "warned": "warned_about",
    "scrutinize": "warned_about",
    "lack": "lacks_validation",
    "lacks": "lacks_validation",
    "responded": "responded_to",
    "audit": "will_audit",
    "cited": "cited_evidence",
    "paused": "paused_initiative",
    "halted": "paused_initiative",
    "confirmed": "confirmed_status",
    "endorsed": "endorsed_proposal",
    "joined": "joined_group",
    "reduces": "targets_outcome",
}

INITIATIVE_PREDICATES = {
    "announced_initiative",
    "paused_initiative",
    "will_audit",
    "lacks_validation",
    "cited_evidence",
    "confirmed_status",
}

CONFLICT_PAIRS: list[tuple[str, str]] = [
    ("announced_initiative", "paused_initiative"),
    ("announced_initiative", "lacks_validation"),
]


def _compile_entity_patterns() -> list[tuple[str, str, re.Pattern[str]]]:
    compiled: list[tuple[str, str, re.Pattern[str]]] = []
    for canonical, (entity_type, aliases) in ENTITY_ALIASES.items():
        ordered_aliases = sorted(aliases, key=len, reverse=True)
        alternation = "|".join(re.escape(alias) for alias in ordered_aliases)
        pattern = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE | re.UNICODE)
        compiled.append((canonical, entity_type, pattern))
    return compiled


ENTITY_PATTERNS = _compile_entity_patterns()

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def get_font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = (
        "arial.ttf",
        "segoeui.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, size)
            _FONT_CACHE[size] = font
            return font
        except Exception:
            continue

    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_demo_documents() -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for item in _DEMO_DOCS:
        text = item["text"]
        documents.append(
            SourceDocument(
                doc_id=item["doc_id"],
                title=item["title"],
                text=text,
                source=item["source"],
                published=item["published"],
                sha256=sha256_text(text),
            )
        )
    return documents


def extract_mentions(documents: list[SourceDocument], settings: PipelineSettings) -> list[Mention]:
    mentions: list[Mention] = []

    for doc in documents:
        text = doc.text

        for canonical, entity_type, pattern in ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                start, end = match.span()

                if matched_text.lower() == canonical.lower():
                    confidence = 0.94
                elif len(matched_text) >= 8:
                    confidence = 0.86
                else:
                    confidence = 0.74

                if entity_type == "PERSON" and "dr" in matched_text.lower():
                    confidence += 0.03

                if entity_type == "ORG" and (
                    "energy" in matched_text.lower()
                    or "grid" in matched_text.lower()
                    or "institute" in matched_text.lower()
                ):
                    confidence += 0.02

                confidence = float(np.clip(confidence, 0.05, 0.99))

                excerpt_start = max(0, start - 60)
                excerpt_end = min(len(text), end + 60)
                excerpt = text[excerpt_start:excerpt_end].strip()

                mention_id = hashlib.sha1(
                    f"{doc.doc_id}:{start}:{end}:{canonical}".encode("utf-8")
                ).hexdigest()[:16]

                mentions.append(
                    Mention(
                        mention_id=mention_id,
                        doc_id=doc.doc_id,
                        entity_id=canonical,
                        entity_label=matched_text,
                        entity_type=entity_type,
                        span_start=start,
                        span_end=end,
                        excerpt=excerpt,
                        extractor_version=settings.extractor_version,
                        confidence=confidence,
                        ingested_at=settings.fixed_ingestion_timestamp,
                    )
                )

    return mentions


def segment_sentences(text: str) -> list[tuple[int, int, str]]:
    if not text.strip():
        return []

    parts = [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    segments: list[tuple[int, int, str]] = []
    cursor = 0

    for part in parts:
        idx = text.find(part, cursor)
        if idx < 0:
            idx = cursor
        start = idx
        end = idx + len(part)
        segments.append((start, end, part))
        cursor = end

    return segments


def extract_relations(
    documents: list[SourceDocument],
    mentions: list[Mention],
    settings: PipelineSettings,
) -> list[Relation]:
    mentions_by_doc: dict[str, list[Mention]] = defaultdict(list)
    for mention in mentions:
        mentions_by_doc[mention.doc_id].append(mention)

    relation_map: dict[str, Relation] = {}
    ingested_at = settings.fixed_ingestion_timestamp

    def add_relation(
        doc_id: str,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float,
        sentence_mentions: list[Mention],
        sentence_span: tuple[int, int, str],
        trigger: str,
    ) -> None:
        if subject_id == object_id:
            return

        key = f"{subject_id}|{predicate}|{object_id}"
        relevant_mention_ids = [
            mention.mention_id
            for mention in sentence_mentions
            if mention.entity_id in {subject_id, object_id}
        ]

        evidence_span = {
            "doc_id": doc_id,
            "start": sentence_span[0],
            "end": sentence_span[1],
            "text": sentence_span[2],
            "trigger": trigger,
        }

        confidence = float(np.clip(confidence, 0.03, 0.97))

        if key not in relation_map:
            relation_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
            relation_map[key] = Relation(
                relation_id=relation_id,
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                confidence=confidence,
                status="candidate",
                evidence_mention_ids=sorted(set(relevant_mention_ids)),
                evidence_docs=[doc_id],
                evidence_spans=[evidence_span],
                extractor_version=settings.extractor_version,
                ingested_at=ingested_at,
            )
            return

        existing = relation_map[key]
        existing.confidence = float(max(existing.confidence, confidence))

        for mention_id in relevant_mention_ids:
            if mention_id not in existing.evidence_mention_ids:
                existing.evidence_mention_ids.append(mention_id)

        if doc_id not in existing.evidence_docs:
            existing.evidence_docs.append(doc_id)

        duplicate_span = any(
            span.get("doc_id") == doc_id
            and span.get("start") == sentence_span[0]
            and span.get("trigger") == trigger
            for span in existing.evidence_spans
        )
        if not duplicate_span:
            existing.evidence_spans.append(evidence_span)

    for doc in documents:
        for start, end, sentence in segment_sentences(doc.text):
            sentence_mentions = [
                mention
                for mention in mentions_by_doc[doc.doc_id]
                if mention.span_start >= start and mention.span_end <= end
            ]
            sentence_mentions.sort(key=lambda item: item.span_start)

            ordered_entities: list[str] = []
            seen_entities: set[str] = set()
            entity_types: dict[str, str] = {}

            for mention in sentence_mentions:
                if mention.entity_id not in seen_entities:
                    ordered_entities.append(mention.entity_id)
                    seen_entities.add(mention.entity_id)
                    entity_types[mention.entity_id] = mention.entity_type

            if len(ordered_entities) < 2:
                continue

            lower_sentence = sentence.lower()

            persons = [entity for entity in ordered_entities if entity_types[entity] == "PERSON"]
            orgs = [entity for entity in ordered_entities if entity_types[entity] == "ORG"]

            if persons and orgs and re.search(r"\b(?:at|with|for)\b", lower_sentence):
                add_relation(
                    doc_id=doc.doc_id,
                    subject_id=persons[0],
                    predicate="affiliated_with",
                    object_id=orgs[0],
                    confidence=0.78,
                    sentence_mentions=sentence_mentions,
                    sentence_span=(start, end, sentence),
                    trigger="affiliation",
                )

            for phrase, predicate in RELATION_TRIGGERS.items():
                if not re.search(rf"\b{re.escape(phrase)}\b", lower_sentence):
                    continue

                if predicate in INITIATIVE_PREDICATES:
                    subject = next(
                        (entity for entity in ordered_entities if entity_types[entity] == "ORG"),
                        ordered_entities[0],
                    )
                    if entity_types.get(subject) not in {"ORG", "PERSON"}:
                        continue

                    candidates = [entity for entity in ordered_entities if entity != subject]
                    object_entity = next(
                        (entity for entity in candidates if entity_types[entity] == "PRODUCT"),
                        candidates[-1] if candidates else ordered_entities[-1],
                    )
                elif predicate in {
                    "responded_to",
                    "warned_about",
                    "endorsed_proposal",
                    "joined_group",
                    "targets_outcome",
                }:
                    subject = ordered_entities[0]
                    object_entity = (
                        ordered_entities[-1]
                        if ordered_entities[-1] != subject
                        else ordered_entities[1]
                    )
                else:
                    subject = ordered_entities[0]
                    object_entity = ordered_entities[-1]

                if subject == object_entity:
                    continue

                subject_mention = next(
                    (mention for mention in sentence_mentions if mention.entity_id == subject),
                    None,
                )
                object_mention = next(
                    (mention for mention in sentence_mentions if mention.entity_id == object_entity),
                    None,
                )

                distance = 200
                if subject_mention is not None and object_mention is not None:
                    distance = abs(object_mention.span_start - subject_mention.span_end)

                confidence = 0.62
                if predicate in {"announced_initiative", "paused_initiative"}:
                    confidence = 0.74
                elif predicate in {"warned_about", "lacks_validation", "confirmed_status"}:
                    confidence = 0.68
                elif predicate == "affiliated_with":
                    confidence = 0.78

                confidence += 0.04
                confidence -= min(0.12, distance / 1800.0)

                add_relation(
                    doc_id=doc.doc_id,
                    subject_id=subject,
                    predicate=predicate,
                    object_id=object_entity,
                    confidence=confidence,
                    sentence_mentions=sentence_mentions,
                    sentence_span=(start, end, sentence),
                    trigger=phrase,
                )

    return list(relation_map.values())


def detect_contradictions(relations: list[Relation]) -> list[Relation]:
    groups: dict[tuple[Any, ...], list[Relation]] = defaultdict(list)

    for relation in relations:
        groups[("pair", relation.subject_id, relation.object_id)].append(relation)
        if relation.predicate in INITIATIVE_PREDICATES:
            groups[("object", relation.object_id)].append(relation)

    def mark_contradicted(relation: Relation, note: str) -> None:
        if relation.status != "contradicted":
            relation.status = "contradicted"
            relation.confidence = float(max(0.05, relation.confidence * 0.65))
        if note not in relation.contradiction_notes:
            relation.contradiction_notes.append(note)

    for grouped_relations in groups.values():
        predicates = {relation.predicate for relation in grouped_relations}

        for left, right in CONFLICT_PAIRS:
            if left in predicates and right in predicates:
                note = f"Conflicting evidence detected: {left} vs {right}."
                for relation in grouped_relations:
                    if relation.predicate in {left, right}:
                        mark_contradicted(relation, note)

    return relations


def build_entity_meta(mentions: list[Mention]) -> dict[str, dict[str, str]]:
    entity_meta: dict[str, dict[str, str]] = {}

    for mention in mentions:
        existing = entity_meta.get(mention.entity_id)
        if existing is None or len(mention.entity_label) > len(existing["label"]):
            entity_meta[mention.entity_id] = {
                "label": mention.entity_label,
                "type": mention.entity_type,
            }

    return entity_meta


def build_graph(relations: list[Relation], entity_meta: dict[str, dict[str, str]]) -> nx.DiGraph:
    graph = nx.DiGraph()

    for entity_id, meta in entity_meta.items():
        graph.add_node(entity_id, label=meta["label"], type=meta["type"])

    for relation in relations:
        for node_id in (relation.subject_id, relation.object_id):
            if node_id not in graph:
                graph.add_node(node_id, label=node_id, type="UNKNOWN")

        if graph.has_edge(relation.subject_id, relation.object_id):
            edge = graph.edges[relation.subject_id, relation.object_id]
            edge["relation_ids"].append(relation.relation_id)
            edge["confidence"] = max(float(edge["confidence"]), float(relation.confidence))
            edge["statuses"].append(relation.status)
        else:
            graph.add_edge(
                relation.subject_id,
                relation.object_id,
                relation_ids=[relation.relation_id],
                predicate=relation.predicate,
                confidence=float(relation.confidence),
                statuses=[relation.status],
            )

    return graph


def compute_positions(graph: nx.DiGraph, settings: PipelineSettings) -> dict[str, tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes())): (0.5, 0.5)}

    try:
        raw_pos = nx.spring_layout(
            graph,
            seed=settings.seed,
            k=1.2 / np.sqrt(max(1, graph.number_of_nodes())),
            iterations=80,
        )
    except Exception:
        nodes = list(graph.nodes())
        raw_pos = {
            node: (
                float(np.cos(2 * np.pi * idx / len(nodes))),
                float(np.sin(2 * np.pi * idx / len(nodes))),
            )
            for idx, node in enumerate(nodes)
        }

    xs = [point[0] for point in raw_pos.values()]
    ys = [point[1] for point in raw_pos.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    range_x = max_x - min_x if max_x != min_x else 1.0
    range_y = max_y - min_y if max_y != min_y else 1.0

    return {
        node: (
            float((point[0] - min_x) / range_x),
            float((point[1] - min_y) / range_y),
        )
        for node, point in raw_pos.items()
    }


def run_simulation(relations: list[Relation], settings: PipelineSettings) -> list[dict[str, Any]]:
    rng = np.random.default_rng(settings.seed)
    states = [relation.model_copy(deep=True) for relation in relations]
    history: list[dict[str, Any]] = []

    for step in range(1, settings.simulation_steps + 1):
        for relation in states:
            evidence_doc_count = len(set(relation.evidence_docs))

            if relation.status == "contradicted":
                delta = -0.018 * evidence_doc_count
            elif relation.status == "accepted":
                delta = 0.004
            else:
                delta = 0.014 * evidence_doc_count

            noise = float(rng.normal(0.0, 0.0035))
            relation.confidence = float(np.clip(relation.confidence + delta + noise, 0.03, 0.98))

            if relation.status != "contradicted":
                if relation.confidence >= 0.76:
                    relation.status = "accepted"
                elif relation.confidence >= settings.min_confidence:
                    relation.status = "review"
                else:
                    relation.status = "candidate"

        confidences = [relation.confidence for relation in states]
        average_confidence = float(np.mean(confidences)) if confidences else 0.0

        history.append(
            {
                "step": step,
                "avg_confidence": average_confidence,
                "accepted": sum(1 for relation in states if relation.status == "accepted"),
                "review": sum(1 for relation in states if relation.status == "review"),
                "candidate": sum(1 for relation in states if relation.status == "candidate"),
                "contradicted": sum(1 for relation in states if relation.status == "contradicted"),
                "relations": [relation.model_copy(deep=True) for relation in states],
            }
        )

    return history


def render_frame(
    state: dict[str, Any],
    settings: PipelineSettings,
    positions: dict[str, tuple[float, float]],
    entity_meta: dict[str, dict[str, str]],
    degrees: dict[str, int],
    frame_index: int,
    total_frames: int,
) -> np.ndarray:
    width = settings.canvas_width
    height = settings.canvas_height

    image = Image.new("RGB", (width, height), (11, 15, 23))
    draw = ImageDraw.Draw(image)

    title_font = get_font(17)
    small_font = get_font(11)
    tiny_font = get_font(10)

    margin_x = 70
    margin_y = 58
    bottom_reserve = 55

    def node_xy(node_id: str) -> tuple[int, int]:
        point = positions.get(node_id, (0.5, 0.5))
        x = int(margin_x + point[0] * (width - 2 * margin_x))
        y = int(margin_y + point[1] * (height - margin_y - bottom_reserve))
        return x, y

    draw.rectangle((0, 0, width, 38), fill=(18, 25, 38))
    draw.text(
        (12, 10),
        "Cross-Document Relationship Intelligence — Live Simulation",
        fill=(236, 240, 241),
        font=title_font,
    )
    draw.text(
        (max(12, width - 175), 12),
        f"step {state['step']}/{settings.simulation_steps}",
        fill=(170, 180, 190),
        font=small_font,
    )

    for edge_index, relation in enumerate(state["relations"]):
        p1 = node_xy(relation.subject_id)
        p2 = node_xy(relation.object_id)
        color = STATUS_COLORS.get(relation.status, (120, 130, 140))
        line_width = 1 + int(relation.confidence * 3)

        draw.line((p1, p2), fill=color, width=line_width)

        pulse_t = ((frame_index / max(1, total_frames - 1)) + edge_index * 0.17) % 1.0
        px = p1[0] + (p2[0] - p1[0]) * pulse_t
        py = p1[1] + (p2[1] - p1[1]) * pulse_t
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(255, 255, 255))

    for node_id, point in positions.items():
        x, y = node_xy(node_id)
        meta = entity_meta.get(node_id, {"label": node_id, "type": "UNKNOWN"})
        degree = degrees.get(node_id, 1)
        radius = 8 + min(10, degree * 2)
        color = TYPE_COLORS.get(meta["type"], (160, 160, 160))

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(240, 240, 240),
            width=1,
        )

        label = meta["label"]
        if len(label) > 20:
            label = label[:19] + "…"

        bbox = draw.textbbox((0, 0), label, font=tiny_font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (int(x - text_width / 2), y + radius + 3),
            label,
            fill=(220, 225, 230),
            font=tiny_font,
        )

    panel_x = 10
    panel_y = height - 92
    panel_w = 235
    panel_h = 82

    try:
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
            radius=8,
            fill=(20, 28, 40),
            outline=(60, 75, 95),
        )
    except Exception:
        draw.rectangle(
            (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
            fill=(20, 28, 40),
            outline=(60, 75, 95),
        )

    metric_lines = [
        f"avg confidence {state['avg_confidence']:.2f}",
        f"accepted {state['accepted']}",
        f"review {state['review']}",
        f"contradicted {state['contradicted']}",
    ]
    for idx, line in enumerate(metric_lines):
        draw.text(
            (panel_x + 10, panel_y + 8 + idx * 17),
            line,
            fill=(210, 220, 230),
            font=small_font,
        )

    legend_x = width - 155
    legend_y = height - 92
    legend_w = 145
    legend_h = 82

    try:
        draw.rounded_rectangle(
            (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
            radius=8,
            fill=(20, 28, 40),
            outline=(60, 75, 95),
        )
    except Exception:
        draw.rectangle(
            (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
            fill=(20, 28, 40),
            outline=(60, 75, 95),
        )

    legend_items = [
        ("accepted", STATUS_COLORS["accepted"]),
        ("review", STATUS_COLORS["review"]),
        ("candidate", STATUS_COLORS["candidate"]),
        ("contradicted", STATUS_COLORS["contradicted"]),
    ]
    for idx, (label, color) in enumerate(legend_items):
        y = legend_y + 12 + idx * 17
        draw.rectangle((legend_x + 10, y, legend_x + 20, y + 10), fill=color)
        draw.text((legend_x + 26, y - 1), label, fill=(210, 220, 230), font=tiny_font)

    return np.asarray(image, dtype=np.uint8)


def generate_frames(
    history: list[dict[str, Any]],
    settings: PipelineSettings,
    positions: dict[str, tuple[float, float]],
    entity_meta: dict[str, dict[str, str]],
    degrees: dict[str, int],
) -> list[np.ndarray]:
    frame_count = max(2, int(round(settings.video_seconds * settings.video_fps)))
    frames: list[np.ndarray] = []

    if not history:
        return frames

    for frame_index in range(frame_count):
        state_index = min(len(history) - 1, int(frame_index * len(history) / frame_count))
        frames.append(
            render_frame(
                state=history[state_index],
                settings=settings,
                positions=positions,
                entity_meta=entity_meta,
                degrees=degrees,
                frame_index=frame_index,
                total_frames=frame_count,
            )
        )

    return frames


def encode_video(frames: list[np.ndarray], settings: PipelineSettings) -> tuple[bytes, str]:
    if not frames:
        return b"", ""

    tmp_path: str | None = None

    try:
        import imageio

        with tempfile.NamedTemporaryFile(prefix="cdri_", suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        imageio.mimwrite(
            tmp_path,
            frames,
            format="FFMPEG",
            codec="libx264",
            fps=settings.video_fps,
            quality=8,
            pixelformat="yuv420p",
        )

        with open(tmp_path, "rb") as fh:
            data = fh.read()

        if data:
            return data, "video/mp4"

    except Exception as exc:
        log_event("mp4_encoding_failed", error=repr(exc))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    try:
        buffer = io.BytesIO()
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(20, int(1000 / settings.video_fps)),
            loop=0,
            optimize=False,
        )
        return buffer.getvalue(), "image/gif"
    except Exception as exc:
        log_event("gif_encoding_failed", error=repr(exc))
        return b"", ""


def build_pipeline(settings: PipelineSettings) -> dict[str, Any]:
    documents = load_demo_documents()
    mentions = extract_mentions(documents, settings)
    relations = extract_relations(documents, mentions, settings)
    relations = detect_contradictions(relations)

    entity_meta = build_entity_meta(mentions)
    graph = build_graph(relations, entity_meta)
    positions = compute_positions(graph, settings)
    degrees = dict(graph.degree())

    history = run_simulation(relations, settings)
    frames = generate_frames(history, settings, positions, entity_meta, degrees)

    if settings.auto_generate_video:
        video_bytes, video_mime = encode_video(frames, settings)
    else:
        video_bytes, video_mime = b"", ""

    documents_df = pd.DataFrame([document.model_dump() for document in documents])
    mentions_df = pd.DataFrame([mention.model_dump() for mention in mentions])
    history_df = pd.DataFrame(
        [{key: value for key, value in state.items() if key != "relations"} for state in history]
    )

    log_event(
        "pipeline_built",
        config_hash=settings.config_hash(),
        documents=len(documents),
        mentions=len(mentions),
        relations=len(relations),
        frames=len(frames),
        video_mime=video_mime,
    )

    return {
        "settings": settings,
        "documents": documents,
        "documents_df": documents_df,
        "documents_by_id": {document.doc_id: document for document in documents},
        "mentions": mentions,
        "mentions_df": mentions_df,
        "relations": relations,
        "entity_meta": entity_meta,
        "graph": graph,
        "positions": positions,
        "degrees": degrees,
        "history": history,
        "history_df": history_df,
        "frames": frames,
        "video_bytes": video_bytes,
        "video_mime": video_mime,
    }


def highlight_span(text: str, start: int, end: int) -> str:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))

    prefix = html.escape(text[:start])
    middle = html.escape(text[start:end])
    suffix = html.escape(text[end:])

    return (
        "<div style='background:#111827; color:#e5e7eb; padding:12px; "
        "border-radius:10px; white-space:pre-wrap; font-family:monospace; "
        "border:1px solid #374151;'>"
        f"{prefix}<mark style='background:#facc15; color:#111827; padding:0 2px; border-radius:3px;'>{middle}</mark>{suffix}"
        "</div>"
    )


def show_image(image: Any, caption: str | None = None) -> None:
    """
    Compatibility wrapper for Streamlit image rendering.

    Avoids passing unsupported keyword arguments such as format=...
    """
    try:
        if caption is None:
            st.image(image, use_container_width=True)
        else:
            st.image(image, caption=caption, use_container_width=True)
        return
    except (TypeError, ValueError):
        pass

    try:
        if caption is None:
            st.image(image, width="stretch")
        else:
            st.image(image, caption=caption, width="stretch")
        return
    except (TypeError, ValueError):
        pass

    if caption is None:
        st.image(image)
    else:
        st.image(image, caption=caption)


def show_dataframe(df: pd.DataFrame, hide_index: bool = True) -> None:
    """
    Compatibility wrapper for Streamlit dataframe rendering.
    """
    try:
        st.dataframe(df, hide_index=hide_index, use_container_width=True)
        return
    except (TypeError, ValueError):
        pass

    try:
        st.dataframe(df, hide_index=hide_index, width="stretch")
        return
    except (TypeError, ValueError):
        pass

    st.dataframe(df, hide_index=hide_index)


def render_embedded_media(video_bytes: bytes, video_mime: str) -> bool:
    """
    Render generated MP4 or GIF media using embedded HTML/base64.

    This avoids Streamlit version-specific issues with st.video/st.image
    keyword arguments.
    """
    if not video_bytes or not video_mime:
        return False

    encoded = base64.b64encode(video_bytes).decode("ascii")
    style = (
        "width:100%; max-width:100%; border-radius:12px; "
        "background:#000; display:block; border:1px solid #374151;"
    )

    if video_mime == "video/mp4":
        st.markdown(
            f'<video autoplay muted loop playsinline controls style="{style}">'
            f'<source src="data:video/mp4;base64,{encoded}" type="video/mp4">'
            "Your browser does not support embedded MP4 video."
            "</video>",
            unsafe_allow_html=True,
        )
        return True

    if video_mime == "image/gif":
        st.markdown(
            f'<img src="data:image/gif;base64,{encoded}" '
            f'alt="Animated cross-document relationship demo" style="{style}" />',
            unsafe_allow_html=True,
        )
        return True

    return False


def sidebar_settings() -> PipelineSettings:
    st.sidebar.title("⚙️ Demo controls")

    min_confidence = st.sidebar.slider(
        "Relation review threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.01,
        help="Relations below this confidence remain candidates unless contradicted.",
    )

    simulation_steps = st.sidebar.slider(
        "Simulation steps",
        min_value=1,
        max_value=30,
        value=8,
        step=1,
        help="Number of deterministic corroboration/contradiction updates.",
    )

    video_fps = st.sidebar.select_slider(
        "Video FPS",
        options=[8, 10, 12, 15, 20],
        value=12,
    )

    video_seconds = st.sidebar.slider(
        "Video length seconds",
        min_value=1.0,
        max_value=5.0,
        value=2.5,
        step=0.5,
    )

    auto_generate_video = st.sidebar.checkbox(
        "Auto-generate live demo video",
        value=True,
        help="Generates an MP4 locally and embeds it. Falls back to GIF if needed.",
    )

    if st.sidebar.button("Rebuild demo"):
        st.session_state.pop("pipeline", None)
        st.session_state.pop("config_hash", None)

    st.sidebar.caption(
        "CPU-only prototype. No external API keys. Demo data is synthetic and clearly labelled."
    )

    return PipelineSettings(
        min_confidence=float(min_confidence),
        simulation_steps=int(simulation_steps),
        video_fps=int(video_fps),
        video_seconds=float(video_seconds),
        auto_generate_video=bool(auto_generate_video),
    )


def render_relation_detail(
    relation: Relation,
    documents_by_id: dict[str, SourceDocument],
    entity_meta: dict[str, dict[str, str]],
) -> None:
    col_subject, col_predicate, col_object = st.columns(3)

    col_subject.metric(
        "Subject",
        entity_meta.get(relation.subject_id, {}).get("label", relation.subject_id),
    )
    col_predicate.metric("Predicate", relation.predicate.replace("_", " "))
    col_object.metric(
        "Object",
        entity_meta.get(relation.object_id, {}).get("label", relation.object_id),
    )

    st.write(
        f"Confidence: **{relation.confidence:.2f}** | "
        f"Status: **{relation.status}** | "
        f"Extractor: `{relation.extractor_version}` | "
        f"Ingested: `{relation.ingested_at}`"
    )

    if relation.contradiction_notes:
        st.error("Contradiction detected: " + "; ".join(relation.contradiction_notes))

    st.subheader("Exact supporting spans")

    if not relation.evidence_spans:
        st.info("No evidence spans recorded for this relation.")
        return

    for index, span in enumerate(relation.evidence_spans, start=1):
        document = documents_by_id.get(str(span.get("doc_id", "")))
        if document is None:
            continue

        st.markdown(
            f"**Evidence {index}** — `{span.get('doc_id')}` "
            f"chars {span.get('start')}–{span.get('end')} "
            f"trigger `{span.get('trigger', '')}`"
        )
        st.markdown(
            highlight_span(document.text, int(span.get("start", 0)), int(span.get("end", 0))),
            unsafe_allow_html=True,
        )


def render_ui(pipeline: dict[str, Any], settings: PipelineSettings) -> None:
    st.title("🕸️ Cross-Document Relationship Intelligence — Working Demo")
    st.caption(
        f"CPU-only • No API key • seed `{settings.seed}` • config `{settings.config_hash()}` • "
        f"extractor `{settings.extractor_version}`"
    )

    media_rendered = render_embedded_media(
        pipeline["video_bytes"],
        pipeline["video_mime"],
    )

    if not media_rendered:
        if pipeline["frames"]:
            show_image(pipeline["frames"][0])
            st.info("Local video encoding unavailable. Showing first simulation frame instead.")
        else:
            st.warning("No simulation frames were generated.")

    if pipeline["video_bytes"]:
        suffix = "mp4" if pipeline["video_mime"] == "video/mp4" else "gif"
        st.download_button(
            "Download demo media",
            data=pipeline["video_bytes"],
            file_name=f"cdri_demo.{suffix}",
            mime=pipeline["video_mime"] or "application/octet-stream",
        )

    if not pipeline["history"]:
        st.warning("Simulation history is empty.")
        return

    latest = pipeline["history"][-1]

    metric_cols = st.columns(6)
    metric_cols[0].metric("Documents", len(pipeline["documents"]))
    metric_cols[1].metric("Mentions", len(pipeline["mentions"]))
    metric_cols[2].metric("Relations", len(latest["relations"]))
    metric_cols[3].metric("Accepted", latest["accepted"])
    metric_cols[4].metric("Review", latest["review"])
    metric_cols[5].metric("Contradicted", latest["contradicted"])

    tabs = st.tabs(
        [
            "Documents",
            "Mentions",
            "Relations & Evidence",
            "Graph Query",
            "Simulation",
        ]
    )

    with tabs[0]:
        documents_df = pipeline["documents_df"]
        show_dataframe(
            documents_df[["doc_id", "title", "source", "published", "sha256"]]
        )

        doc_id = st.selectbox("Select document", documents_df["doc_id"].tolist())
        document = pipeline["documents_by_id"][doc_id]

        st.markdown(f"**{document.title}** — {document.source}, {document.published}")
        st.text_area(
            "Read-only raw text",
            document.text,
            height=240,
            disabled=True,
            label_visibility="collapsed",
        )

    with tabs[1]:
        mentions_df = pipeline["mentions_df"].copy()
        entity_options = sorted(pipeline["entity_meta"].keys())
        document_options = ["All"] + pipeline["documents_df"]["doc_id"].tolist()

        selected_entity = st.selectbox("Filter by entity", ["All"] + entity_options)
        selected_document = st.selectbox("Filter by document", document_options)

        filtered_mentions = mentions_df
        if selected_entity != "All":
            filtered_mentions = filtered_mentions[filtered_mentions["entity_id"] == selected_entity]
        if selected_document != "All":
            filtered_mentions = filtered_mentions[filtered_mentions["doc_id"] == selected_document]

        show_dataframe(
            filtered_mentions[
                [
                    "mention_id",
                    "doc_id",
                    "entity_label",
                    "entity_type",
                    "span_start",
                    "span_end",
                    "confidence",
                    "excerpt",
                ]
            ]
        )

    with tabs[2]:
        relations = latest["relations"]

        if not relations:
            st.info("No relations were extracted from the demo corpus.")
        else:
            relation_rows = [
                relation.model_dump(exclude={"evidence_spans"}) for relation in relations
            ]
            relations_df = pd.DataFrame(relation_rows)

            status_options = sorted(relations_df["status"].unique().tolist())
            selected_statuses = st.multiselect(
                "Status filter",
                options=status_options,
                default=status_options,
            )

            display_min_confidence = st.slider(
                "Minimum displayed confidence",
                min_value=0.0,
                max_value=1.0,
                value=float(settings.min_confidence),
                step=0.01,
                key="relation_display_min_confidence",
            )

            filtered_relations = relations_df[
                relations_df["status"].isin(selected_statuses)
                & (relations_df["confidence"] >= display_min_confidence)
            ]

            show_dataframe(
                filtered_relations[
                    [
                        "relation_id",
                        "subject_id",
                        "predicate",
                        "object_id",
                        "confidence",
                        "status",
                        "evidence_docs",
                    ]
                ]
            )

            if filtered_relations.empty:
                st.warning("No relations match the current filters.")
            else:
                selected_relation_id = st.selectbox(
                    "Inspect relation",
                    filtered_relations["relation_id"].tolist(),
                )
                selected_relation = next(
                    relation
                    for relation in relations
                    if relation.relation_id == selected_relation_id
                )
                render_relation_detail(
                    selected_relation,
                    pipeline["documents_by_id"],
                    pipeline["entity_meta"],
                )

    with tabs[3]:
        graph = pipeline["graph"]
        node_list = sorted(pipeline["entity_meta"].keys())

        if not node_list:
            st.info("No entities are available for graph exploration.")
        else:
            source_col, target_col = st.columns(2)

            source_index = 0
            target_index = 1 if len(node_list) > 1 else 0

            source_entity = source_col.selectbox("Source entity", node_list, index=source_index)
            target_entity = target_col.selectbox("Target entity", node_list, index=target_index)

            if source_entity != target_entity:
                undirected_graph = graph.to_undirected()
                try:
                    path = nx.shortest_path(undirected_graph, source_entity, target_entity)
                    readable_path = [
                        pipeline["entity_meta"].get(node, {}).get("label", node) for node in path
                    ]
                    st.success("Path: " + " → ".join(readable_path))
                except nx.NetworkXNoPath:
                    st.info("No undirected path found between the selected entities.")
                except nx.NodeNotFound:
                    st.warning("One of the selected entities is not present in the graph.")
                except Exception as exc:
                    st.warning(f"Path lookup failed: {exc}")
            else:
                st.info("Select two different entities to explore a path.")

            st.subheader("Immediate neighbors")
            neighbors = sorted(
                set(list(graph.successors(source_entity)) + list(graph.predecessors(source_entity)))
            )

            if neighbors:
                neighbor_labels = [
                    pipeline["entity_meta"].get(node, {}).get("label", node) for node in neighbors
                ]
                st.write(", ".join(neighbor_labels))
            else:
                st.write("No immediate neighbors.")

    with tabs[4]:
        history_df = pipeline["history_df"]

        if history_df.empty:
            st.info("No simulation history available.")
        else:
            chart_df = history_df.set_index("step")
            st.subheader("Average confidence over simulation steps")
            st.line_chart(chart_df[["avg_confidence"]])

            st.subheader("Relation status counts over simulation steps")
            st.bar_chart(chart_df[["accepted", "review", "contradicted"]])

        if pipeline["frames"]:
            st.subheader("Inspect individual simulation frames")
            frame_index = st.slider(
                "Frame",
                min_value=0,
                max_value=len(pipeline["frames"]) - 1,
                value=0,
            )
            show_image(pipeline["frames"][frame_index])

        st.subheader("Latest relation decisions")
        decision_rows: list[dict[str, Any]] = []

        for relation in latest["relations"]:
            top_evidence = ""
            if relation.evidence_spans:
                top_evidence = str(relation.evidence_spans[0].get("text", ""))[:180]

            decision_rows.append(
                {
                    "relation_id": relation.relation_id,
                    "subject": pipeline["entity_meta"]
                    .get(relation.subject_id, {})
                    .get("label", relation.subject_id),
                    "predicate": relation.predicate,
                    "object": pipeline["entity_meta"]
                    .get(relation.object_id, {})
                    .get("label", relation.object_id),
                    "confidence": round(relation.confidence, 3),
                    "status": relation.status,
                    "evidence_docs": ", ".join(relation.evidence_docs),
                    "top_evidence": top_evidence,
                }
            )

        show_dataframe(pd.DataFrame(decision_rows))


def main() -> None:
    st.set_page_config(
        page_title="Cross-Document Relationship Intelligence Demo",
        page_icon="🕸️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        settings = sidebar_settings()
    except Exception as exc:
        st.error("Invalid demo settings.")
        st.exception(exc)
        return

    config_hash = settings.config_hash()

    if st.session_state.get("config_hash") != config_hash or "pipeline" not in st.session_state:
        try:
            with st.spinner(
                "Loading realistic demo data, extracting provenance, "
                "and rendering live simulation video..."
            ):
                st.session_state.pipeline = build_pipeline(settings)
                st.session_state.config_hash = config_hash
        except Exception as exc:
            st.session_state.pop("pipeline", None)
            st.session_state.config_hash = None
            st.error("Demo build failed.")
            st.exception(exc)
            return

    render_ui(st.session_state.pipeline, settings)


if __name__ == "__main__":
    main()